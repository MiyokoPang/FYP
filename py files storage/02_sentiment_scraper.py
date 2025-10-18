# import dependencies
import praw
import requests
import mysql.connector
from datetime import datetime, timedelta
import time
import logging
from textblob import TextBlob
import re

# Configurations
reddit_config = {
    'client_id': '6SD-D4KilOopu4O6m9R9VA',
    'client_secret': 'mv-Z3_vFTC7WApSWvtdxUX0MKDq3QQ',
    'user_agent': 'Trading Sentiment Bot by Delicious_Divide6891'
}

news_api_key = '73e9447f080543c3885ec7803f705101'

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

class SentimentScraper:
    def __init__(self, reddit_config, news_api_key, db_config):
        self.reddit_config = reddit_config
        self.news_api_key = news_api_key
        self.db_config = db_config
        self.setup_logging()
        self.reddit = None
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sentiment_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_db(self):
        """Connect to MySQL database"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except mysql.connector.Error as e:
            self.logger.error(f"Database connection error: {e}")
            return None
    
    def create_sentiment_tables(self):
        """Create tables for sentiment data"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Reddit sentiment data - NO URL COLUMN
        reddit_table = """
        CREATE TABLE IF NOT EXISTS reddit_sentiment (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            subreddit VARCHAR(50),
            post_id VARCHAR(20) UNIQUE,
            title TEXT,
            selftext TEXT,
            score INT,
            num_comments INT,
            created_utc TIMESTAMP,
            sentiment_score DECIMAL(5,4),
            sentiment_label VARCHAR(20),
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_symbol (symbol),
            INDEX idx_created (created_utc)
        )
        """
        
        # News sentiment data - NO URL COLUMN
        news_table = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            source VARCHAR(100),
            title TEXT,
            description TEXT,
            published_at TIMESTAMP,
            sentiment_score DECIMAL(5,4),
            sentiment_label VARCHAR(20),
            scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_symbol (symbol),
            INDEX idx_published (published_at)
        )
        """
        
        # Aggregated daily sentiment
        daily_sentiment_table = """
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            date DATE,
            reddit_avg_sentiment DECIMAL(5,4),
            reddit_post_count INT DEFAULT 0,
            news_avg_sentiment DECIMAL(5,4),
            news_article_count INT DEFAULT 0,
            combined_sentiment DECIMAL(5,4),
            total_mentions INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_symbol_date (symbol, date)
        )
        """
        
        tables = [reddit_table, news_table, daily_sentiment_table]
        
        try:
            for table in tables:
                cursor.execute(table)
            conn.commit()
            self.logger.info("Sentiment tables created successfully")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error creating sentiment tables: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def init_reddit(self):
        """Initialize Reddit API connection"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.reddit_config['client_id'],
                client_secret=self.reddit_config['client_secret'],
                user_agent=self.reddit_config['user_agent']
            )
            self.logger.info("Reddit API initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Reddit API: {e}")
            return False
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using TextBlob
        Returns: (sentiment_score, sentiment_label)
        Score: -1 (very negative) to +1 (very positive)
        """
        if not text or len(text.strip()) == 0:
            return 0.0, 'neutral'
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Classify sentiment
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return round(polarity, 4), label
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, 'neutral'
    
    def scrape_reddit(self, symbols, subreddits=['stocks', 'investing', 'wallstreetbets', 'SecurityAnalysis'], 
                     limit=100, time_filter='week'):
        if not self.reddit:
            if not self.init_reddit():
                return False
        
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        total_posts = 0
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{len(symbols)}] Processing {symbol}...")
            for subreddit_name in subreddits:
                try:
                    print(f"  → Searching r/{subreddit_name}...", end='')
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for stock symbol
                    search_query = f"${symbol} OR {symbol}"
                    
                    post_count = 0
                    for post in subreddit.search(search_query, time_filter=time_filter, limit=limit):
                        post_count += 1
                        # Combine title and selftext for sentiment analysis
                        full_text = f"{post.title} {post.selftext}"
                        sentiment_score, sentiment_label = self.analyze_sentiment(full_text)
                        
                        try:
                            query = """
                            INSERT IGNORE INTO reddit_sentiment 
                            (symbol, subreddit, post_id, title, selftext, score, num_comments, 
                             created_utc, sentiment_score, sentiment_label)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """
                            
                            values = (
                                symbol,
                                subreddit_name,
                                post.id,
                                post.title[:500],  # Limit length
                                post.selftext[:1000] if post.selftext else '',
                                post.score,
                                post.num_comments,
                                datetime.fromtimestamp(post.created_utc),
                                sentiment_score,
                                sentiment_label
                            )
                            
                            cursor.execute(query, values)
                            total_posts += 1
                            
                        except mysql.connector.Error as e:
                            if "Duplicate entry" not in str(e):
                                self.logger.error(f"Error inserting Reddit post: {e}")
                    
                    print(f" {post_count} posts found")
                    # Rate limiting 
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error scraping r/{subreddit_name} for {symbol}: {e}")
                    continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.logger.info(f"Scraped {total_posts} Reddit posts")
        return True
    
    def scrape_news(self, symbols, days_back=7):
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        total_articles = 0
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        for idx, symbol in enumerate(symbols, 1):
            print(f"\n[{idx}/{len(symbols)}] Fetching news for {symbol}...")
            try:
                # NewsAPI endpoint
                url = 'https://newsapi.org/v2/everything'
                
                params = {
                    'q': f"{symbol} stock OR {symbol} shares",
                    'from': from_date.strftime('%Y-%m-%d'),
                    'to': to_date.strftime('%Y-%m-%d'),
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'apiKey': self.news_api_key,
                    'pageSize': 100
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        # Analyze sentiment from title + description
                        text = f"{article.get('title', '')} {article.get('description', '')}"
                        sentiment_score, sentiment_label = self.analyze_sentiment(text)
                        
                        try:
                            query = """
                            INSERT IGNORE INTO news_sentiment
                            (symbol, source, title, description, published_at, 
                             sentiment_score, sentiment_label)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """
                            
                            # Parse published date
                            published = article.get('publishedAt', '')
                            if published:
                                published_dt = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
                            else:
                                published_dt = datetime.now()
                            
                            values = (
                                symbol,
                                article.get('source', {}).get('name', 'Unknown')[:100],
                                article.get('title', '')[:500],
                                article.get('description', '')[:1000],
                                published_dt,
                                sentiment_score,
                                sentiment_label
                            )
                            
                            cursor.execute(query, values)
                            total_articles += 1
                            
                        except mysql.connector.Error as e:
                            if "Duplicate entry" not in str(e):
                                self.logger.error(f"Error inserting news article: {e}")
                    
                    print(f"  ✓ Found {len(articles)} articles")
                    
                elif response.status_code == 426:
                    self.logger.warning("NewsAPI rate limit reached. Upgrade plan or wait.")
                    break
                else:
                    self.logger.error(f"NewsAPI error for {symbol}: {response.status_code}")
                
                # Rate limiting
                time.sleep(1.5)
                
            except Exception as e:
                self.logger.error(f"Error scraping news for {symbol}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.logger.info(f"Scraped {total_articles} news articles")
        return True
    
    def aggregate_daily_sentiment(self, date=None):
        """
        Aggregate sentiment data by symbol and date
        date: specific date to aggregate (default: today)
        """
        if date is None:
            date = datetime.now().date()
        
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Get list of symbols
        cursor.execute("SELECT DISTINCT symbol FROM stocks")
        symbols = [row[0] for row in cursor.fetchall()]
        
        for symbol in symbols:
            try:
                # Aggregate Reddit sentiment
                reddit_query = """
                SELECT 
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as post_count
                FROM reddit_sentiment
                WHERE symbol = %s AND DATE(created_utc) = %s
                """
                cursor.execute(reddit_query, (symbol, date))
                reddit_result = cursor.fetchone()
                reddit_avg = reddit_result[0] if reddit_result[0] else 0
                reddit_count = reddit_result[1] if reddit_result[1] else 0
                
                # Aggregate News sentiment
                news_query = """
                SELECT 
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as article_count
                FROM news_sentiment
                WHERE symbol = %s AND DATE(published_at) = %s
                """
                cursor.execute(news_query, (symbol, date))
                news_result = cursor.fetchone()
                news_avg = news_result[0] if news_result[0] else 0
                news_count = news_result[1] if news_result[1] else 0
                
                # Calculate combined sentiment (weighted by count)
                total_count = reddit_count + news_count
                if total_count > 0:
                    combined = ((reddit_avg * reddit_count) + (news_avg * news_count)) / total_count
                else:
                    combined = 0
                
                # Insert or update aggregated data
                insert_query = """
                INSERT INTO daily_sentiment 
                (symbol, date, reddit_avg_sentiment, reddit_post_count, 
                 news_avg_sentiment, news_article_count, combined_sentiment, total_mentions)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    reddit_avg_sentiment = VALUES(reddit_avg_sentiment),
                    reddit_post_count = VALUES(reddit_post_count),
                    news_avg_sentiment = VALUES(news_avg_sentiment),
                    news_article_count = VALUES(news_article_count),
                    combined_sentiment = VALUES(combined_sentiment),
                    total_mentions = VALUES(total_mentions)
                """
                
                cursor.execute(insert_query, (
                    symbol, date, reddit_avg, reddit_count,
                    news_avg, news_count, combined, total_count
                ))
                
            except mysql.connector.Error as e:
                self.logger.error(f"Error aggregating sentiment for {symbol}: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.logger.info(f"Aggregated sentiment data for {len(symbols)} symbols on {date}")
        return True
    
    def get_stock_symbols(self):
        """Get all tracked stock symbols from database"""
        conn = self.connect_db()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM stocks")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return symbols
    
# Initialize scraper
scraper = SentimentScraper(reddit_config, news_api_key, db_config)

# Create sentiment tables
scraper.create_sentiment_tables()

# Get stock symbols from database
symbols = scraper.get_stock_symbols()
print(f"✓ Setup complete! Tracking {len(symbols)} stocks: {symbols}")

print("Starting Reddit scraping...")
scraper.scrape_reddit(symbols, limit=50, time_filter='week')
print("\n✓ Reddit scraping complete!")

print("Starting news scraping...")
print("This will take 1-2 minutes.\n")

scraper.scrape_news(symbols, days_back=7)

print("\n✓ News scraping complete!")

print("Aggregating daily sentiment scores...")

scraper.aggregate_daily_sentiment()

print("✓ Sentiment aggregation complete!")

import pandas as pd

conn = mysql.connector.connect(**db_config)

# Reddit posts count
reddit_df = pd.read_sql("SELECT COUNT(*) as total FROM reddit_sentiment", conn)
print(f"Reddit posts collected: {reddit_df['total'][0]}")

# News articles count
news_df = pd.read_sql("SELECT COUNT(*) as total FROM news_sentiment", conn)
print(f"News articles collected: {news_df['total'][0]}")

# Top stocks by sentiment mentions
print("\n" + "="*60)
print("TOP 10 STOCKS BY TOTAL MENTIONS:")
print("="*60)
sentiment_df = pd.read_sql("""
    SELECT symbol, combined_sentiment, total_mentions, 
           reddit_post_count, news_article_count
    FROM daily_sentiment 
    WHERE date = CURDATE()
    ORDER BY total_mentions DESC 
    LIMIT 10
""", conn)

for idx, row in sentiment_df.iterrows():
    print(f"{row['symbol']:6s} | Sentiment: {row['combined_sentiment']:+.3f} | "
          f"Mentions: {row['total_mentions']:3d} "
          f"(Reddit: {row['reddit_post_count']}, News: {row['news_article_count']})")

conn.close()

print("\n✓ All sentiment data collection complete!")
print("Check your database tables: reddit_sentiment, news_sentiment, daily_sentiment")
