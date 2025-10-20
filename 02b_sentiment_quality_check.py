# ========================================
# sentiment_diagnostic.py
# Run this to check if scrapers are collecting NEW data or duplicates
# ========================================

import mysql.connector
import pandas as pd
from datetime import datetime, timedelta

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

conn = mysql.connector.connect(**db_config)

print("="*70)
print("SENTIMENT DATA DIAGNOSTIC")
print("="*70)

# ========================================
# CHECK 1: Date Range Coverage
# ========================================
print("\n1. DATE COVERAGE")
print("-"*70)

query = """
SELECT 
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(DISTINCT date) as unique_days,
    COUNT(*) as total_records
FROM daily_sentiment
"""
df = pd.read_sql(query, conn)
print(f"First sentiment date: {df['first_date'][0]}")
print(f"Last sentiment date:  {df['last_date'][0]}")
print(f"Unique days:          {df['unique_days'][0]}")
print(f"Total records:        {df['total_records'][0]}")
print(f"Avg records/day:      {df['total_records'][0] / df['unique_days'][0]:.1f}")

# ========================================
# CHECK 2: Daily Breakdown (Last 30 Days)
# ========================================
print("\n2. DAILY SENTIMENT BREAKDOWN (Last 30 days)")
print("-"*70)

query = """
SELECT 
    date,
    COUNT(DISTINCT symbol) as stocks_covered,
    SUM(reddit_post_count) as total_reddit_posts,
    SUM(news_article_count) as total_news_articles,
    SUM(total_mentions) as total_mentions,
    AVG(combined_sentiment) as avg_sentiment
FROM daily_sentiment
WHERE date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY date
ORDER BY date DESC
"""
df = pd.read_sql(query, conn)

if not df.empty:
    print(f"{'Date':<12} {'Stocks':<8} {'Reddit':<8} {'News':<8} {'Total':<8} {'Avg Sent'}")
    print("-"*70)
    for _, row in df.iterrows():
        print(f"{str(row['date']):<12} {row['stocks_covered']:<8} "
              f"{row['total_reddit_posts']:<8} {row['total_news_articles']:<8} "
              f"{row['total_mentions']:<8} {row['avg_sentiment']:>+7.3f}")
    
    # Check for duplicates (same count multiple days)
    duplicate_days = df[df.duplicated(subset=['total_reddit_posts', 'total_news_articles'], keep=False)]
    if len(duplicate_days) > 0:
        print("\n⚠️  WARNING: Possible duplicate scraping detected!")
        print(f"Found {len(duplicate_days)} days with identical counts")
    else:
        print("\n✓ No obvious duplicates detected")
else:
    print("No sentiment data in last 30 days")

# ========================================
# CHECK 3: Reddit Posts - Check for Duplicate IDs
# ========================================
print("\n3. REDDIT DUPLICATE CHECK")
print("-"*70)

query = """
SELECT 
    post_id,
    COUNT(*) as duplicate_count,
    MIN(scraped_at) as first_scraped,
    MAX(scraped_at) as last_scraped
FROM reddit_sentiment
GROUP BY post_id
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC
LIMIT 10
"""
df = pd.read_sql(query, conn)

if not df.empty:
    print(f"⚠️  Found {len(df)} duplicate Reddit post_ids")
    print("\nTop duplicates:")
    for _, row in df.iterrows():
        print(f"  post_id: {row['post_id']} - scraped {row['duplicate_count']} times")
else:
    print("✓ No duplicate Reddit posts (post_id is unique)")

# ========================================
# CHECK 4: Reddit Posts by Date Created
# ========================================
print("\n4. REDDIT POSTS - WHEN WERE THEY CREATED?")
print("-"*70)

query = """
SELECT 
    DATE(created_utc) as post_date,
    COUNT(*) as posts,
    COUNT(DISTINCT symbol) as stocks
FROM reddit_sentiment
GROUP BY DATE(created_utc)
ORDER BY post_date DESC
LIMIT 20
"""
df = pd.read_sql(query, conn)

if not df.empty:
    print(f"{'Post Date':<12} {'Posts':<8} {'Stocks'}")
    print("-"*70)
    for _, row in df.iterrows():
        print(f"{str(row['post_date']):<12} {row['posts']:<8} {row['stocks']}")
    
    # Check if all posts are from same date range (indicates scraping same data)
    date_range = (df['post_date'].max() - df['post_date'].min()).days
    print(f"\nPost date range: {date_range} days")
    if date_range < 3:
        print("⚠️  WARNING: All Reddit posts from same 3-day window!")
        print("   This suggests scraper is fetching the same recent posts repeatedly")
else:
    print("No Reddit data")

# ========================================
# CHECK 5: News Articles by Published Date
# ========================================
print("\n5. NEWS ARTICLES - WHEN WERE THEY PUBLISHED?")
print("-"*70)

query = """
SELECT 
    DATE(published_at) as pub_date,
    COUNT(*) as articles,
    COUNT(DISTINCT symbol) as stocks
FROM news_sentiment
GROUP BY DATE(published_at)
ORDER BY pub_date DESC
LIMIT 20
"""
df = pd.read_sql(query, conn)

if not df.empty:
    print(f"{'Pub Date':<12} {'Articles':<8} {'Stocks'}")
    print("-"*70)
    for _, row in df.iterrows():
        print(f"{str(row['pub_date']):<12} {row['articles']:<8} {row['stocks']}")
    
    date_range = (df['pub_date'].max() - df['pub_date'].min()).days
    print(f"\nArticle date range: {date_range} days")
    if date_range < 5:
        print("⚠️  WARNING: All news from same 5-day window!")
        print("   This suggests NewsAPI is returning same articles repeatedly")
else:
    print("No news data")

# ========================================
# CHECK 6: Stocks with Most/Least Coverage
# ========================================
print("\n6. SENTIMENT COVERAGE BY STOCK")
print("-"*70)

query = """
SELECT 
    symbol,
    COUNT(DISTINCT date) as days_with_sentiment,
    SUM(total_mentions) as total_mentions,
    AVG(combined_sentiment) as avg_sentiment
FROM daily_sentiment
GROUP BY symbol
ORDER BY days_with_sentiment DESC
"""
df = pd.read_sql(query, conn)

if not df.empty:
    print(f"{'Symbol':<8} {'Days':<8} {'Mentions':<10} {'Avg Sent'}")
    print("-"*70)
    for _, row in df.head(10).iterrows():
        print(f"{row['symbol']:<8} {row['days_with_sentiment']:<8} "
              f"{row['total_mentions']:<10} {row['avg_sentiment']:>+7.3f}")
    
    print(f"\n... and {len(df)-10} more stocks")
    print(f"\nStocks with NO sentiment: {20 - len(df)}")
else:
    print("No sentiment data at all")

conn.close()

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nINTERPRETATION:")
print("✓ Good: Unique days increasing, different counts each day")
print("⚠️  Bad: Same Reddit post_ids or article counts repeating")
print("⚠️  Bad: All posts from same 3-7 day window (not getting new data)")