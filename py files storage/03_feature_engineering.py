# ========================================
# 03_enhanced_feature_engineering.py
# Purpose: Create training-ready features combining price + sentiment
# Key Improvement: Smart sentiment handling (no fake neutral padding)
# ========================================

import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CELL 1: Database Configuration
# ========================================

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

print("✓ Imports loaded")
print("✓ Database config ready")


# ========================================
# CELL 2: Enhanced Feature Engineering Class
# ========================================

class EnhancedFeatureEngineering:
    def __init__(self, db_config):
        self.db_config = db_config
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_feature_engineering.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect_db(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except mysql.connector.Error as e:
            self.logger.error(f"Database connection error: {e}")
            return None
    
    def load_price_data(self, symbol, start_date=None, end_date=None):
        """Load historical price data for a symbol"""
        conn = self.connect_db()
        if not conn:
            return None
        
        query = """
        SELECT date, open_price, high_price, low_price, close_price, volume
        FROM historical_prices
        WHERE symbol = %s
        """
        
        params = [symbol]
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        query += " ORDER BY date ASC"
        
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        
        if df.empty:
            self.logger.warning(f"No price data found for {symbol}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def load_sentiment_data(self, symbol):
        """Load sentiment data for a symbol"""
        conn = self.connect_db()
        if not conn:
            return None
        
        query = """
        SELECT date, 
               reddit_avg_sentiment,
               reddit_post_count,
               news_avg_sentiment,
               news_article_count,
               combined_sentiment,
               total_mentions
        FROM daily_sentiment
        WHERE symbol = %s
        ORDER BY date ASC
        """
        
        df = pd.read_sql(query, conn, params=[symbol])
        conn.close()
        
        if df.empty:
            self.logger.warning(f"No sentiment data found for {symbol}")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return df
    
    def merge_price_sentiment(self, price_df, sentiment_df, strategy='forward_fill'):
        """
        Merge price and sentiment data with smart missing value handling
        
        Strategies:
        - 'forward_fill': Use last known sentiment (assumes persistence) - RECOMMENDED
        - 'drop_missing': Only keep dates with real sentiment (cleanest but smallest dataset)
        - 'neutral_zero': Fill with 0 (old method, NOT RECOMMENDED)
        - 'missing_indicator': Use -999 for missing (model learns to ignore it)
        """
        if sentiment_df is None or sentiment_df.empty:
            self.logger.warning("No sentiment data - creating neutral baseline")
            # Create neutral sentiment columns
            price_df['reddit_avg_sentiment'] = 0.0
            price_df['reddit_post_count'] = 0
            price_df['news_avg_sentiment'] = 0.0
            price_df['news_article_count'] = 0
            price_df['combined_sentiment'] = 0.0
            price_df['total_mentions'] = 0
            price_df['has_sentiment'] = 0
            return price_df
        
        # Merge on date
        merged = price_df.merge(sentiment_df, left_index=True, right_index=True, how='left')
        
        sentiment_cols = ['reddit_avg_sentiment', 'news_avg_sentiment', 'combined_sentiment']
        count_cols = ['reddit_post_count', 'news_article_count', 'total_mentions']
        
        # Apply selected strategy
        if strategy == 'forward_fill':
            # Forward fill sentiment (last known value persists)
            for col in sentiment_cols:
                merged[col] = merged[col].fillna(method='ffill').fillna(0.0)
            for col in count_cols:
                merged[col] = merged[col].fillna(0)
                
        elif strategy == 'drop_missing':
            # Keep only days with real sentiment
            merged = merged.dropna(subset=['combined_sentiment'])
            self.logger.info(f"Dropped rows without sentiment. Remaining: {len(merged)} days")
            
        elif strategy == 'missing_indicator':
            # Use -999 for missing sentiment
            for col in sentiment_cols:
                merged[col] = merged[col].fillna(-999.0)
            for col in count_cols:
                merged[col] = merged[col].fillna(0)
                
        else:  # 'neutral_zero' (old method)
            for col in sentiment_cols:
                merged[col] = merged[col].fillna(0.0)
            for col in count_cols:
                merged[col] = merged[col].fillna(0)
        
        # Flag for whether sentiment data exists
        merged['has_sentiment'] = (merged['total_mentions'] > 0).astype(int)
        
        return merged
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators from OHLCV data"""
        if df is None or df.empty:
            return None
        
        df = df.copy()
        
        # Moving Averages
        df['MA_5'] = df['close_price'].rolling(window=5).mean()
        df['MA_10'] = df['close_price'].rolling(window=10).mean()
        df['MA_20'] = df['close_price'].rolling(window=20).mean()
        df['MA_50'] = df['close_price'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['close_price'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close_price'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # RSI (Relative Strength Index)
        delta = df['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['close_price'].rolling(window=20).mean()
        bb_std = df['close_price'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['close_price'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price momentum
        df['momentum_1'] = df['close_price'].pct_change(1)
        df['momentum_5'] = df['close_price'].pct_change(5)
        df['momentum_10'] = df['close_price'].pct_change(10)
        df['momentum_20'] = df['close_price'].pct_change(20)
        
        # Volatility
        df['volatility_5'] = df['close_price'].rolling(window=5).std()
        df['volatility_20'] = df['close_price'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # Volume indicators
        df['volume_MA_5'] = df['volume'].rolling(window=5).mean()
        df['volume_MA_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_MA_20']
        
        # Price range
        df['daily_range'] = df['high_price'] - df['low_price']
        df['daily_return'] = df['close_price'].pct_change()
        
        # Average True Range (ATR)
        high_low = df['high_price'] - df['low_price']
        high_close = np.abs(df['high_price'] - df['close_price'].shift())
        low_close = np.abs(df['low_price'] - df['close_price'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['close_price'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek  # 0=Monday, 4=Friday
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        
        return df
    
    def calculate_sentiment_features(self, df):
        """Calculate additional sentiment-based features"""
        if df is None or df.empty:
            return None
        
        df = df.copy()
        
        # Sentiment momentum (change over time)
        df['sentiment_momentum_3'] = df['combined_sentiment'].diff(3)
        df['sentiment_momentum_7'] = df['combined_sentiment'].diff(7)
        
        # Sentiment moving averages
        df['sentiment_MA_3'] = df['combined_sentiment'].rolling(window=3).mean()
        df['sentiment_MA_7'] = df['combined_sentiment'].rolling(window=7).mean()
        
        # Mention volume trends
        df['mentions_MA_3'] = df['total_mentions'].rolling(window=3).mean()
        df['mentions_trend'] = df['total_mentions'] / (df['mentions_MA_3'] + 1)  # +1 to avoid div by 0
        
        # Sentiment-price correlation features
        df['sentiment_price_alignment'] = np.sign(df['combined_sentiment']) * np.sign(df['daily_return'])
        
        # Reddit vs News sentiment divergence
        df['sentiment_divergence'] = df['reddit_avg_sentiment'] - df['news_avg_sentiment']
        
        return df
    
    def create_supervised_dataset(self, df, lookback=60, forecast_horizon=1, predict_change=True):
        """
        Create supervised learning dataset
        predict_change: If True, predict price change %. If False, predict absolute price.
        """
        if df is None or len(df) < lookback + forecast_horizon:
            self.logger.warning("Insufficient data for supervised dataset creation")
            return None, None, None, None
        
        # Drop rows with NaN values from technical indicators
        df = df.dropna()
        
        if len(df) < lookback + forecast_horizon:
            self.logger.warning("Insufficient data after dropping NaN values")
            return None, None, None, None
        
        # Define feature columns
        price_features = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_diff',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
            'momentum_1', 'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_5', 'volatility_20', 'volatility_ratio',
            'volume_MA_5', 'volume_MA_20', 'volume_ratio',
            'daily_range', 'daily_return', 'ATR', 'OBV',
            'day_of_week', 'month', 'quarter', 'day_of_month'
        ]
        
        sentiment_features = [
            'reddit_avg_sentiment', 'reddit_post_count',
            'news_avg_sentiment', 'news_article_count',
            'combined_sentiment', 'total_mentions', 'has_sentiment',
            'sentiment_momentum_3', 'sentiment_momentum_7',
            'sentiment_MA_3', 'sentiment_MA_7',
            'mentions_MA_3', 'mentions_trend',
            'sentiment_price_alignment', 'sentiment_divergence'
        ]
        
        # Check which features exist in dataframe
        available_features = [col for col in price_features + sentiment_features if col in df.columns]
        
        X_list = []
        y_list = []
        dates_list = []
        current_prices = []
        
        for i in range(lookback, len(df) - forecast_horizon + 1):
            # Features: past 'lookback' days
            X_window = df[available_features].iloc[i-lookback:i].values
            
            # Flatten the window into a single feature vector
            X_flat = X_window.flatten()
            
            # Current price (for calculating % change)
            current_price = df['close_price'].iloc[i-1]
            
            # Target price
            future_price = df['close_price'].iloc[i + forecast_horizon - 1]
            
            if predict_change:
                # Target: percentage price change
                y_target = ((future_price - current_price) / current_price) * 100
            else:
                # Target: absolute price
                y_target = future_price
            
            X_list.append(X_flat)
            y_list.append(y_target)
            dates_list.append(df.index[i + forecast_horizon - 1])
            current_prices.append(current_price)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y, dates_list, current_prices
    
    def prepare_training_data(self, symbol, lookback=60, test_size=0.2, 
                            predict_change=True, sentiment_strategy='forward_fill'):
        """
        Complete pipeline with sentiment integration
        
        sentiment_strategy options:
        - 'forward_fill': Last known sentiment persists (DEFAULT - RECOMMENDED)
        - 'drop_missing': Only use days with real sentiment (cleanest but tiny dataset)
        - 'missing_indicator': Use -999 for missing
        - 'neutral_zero': Old method (not recommended)
        """
        self.logger.info(f"Preparing enhanced training data for {symbol}")
        
        # Load price data
        price_df = self.load_price_data(symbol)
        
        if price_df is None or price_df.empty:
            self.logger.error(f"No price data available for {symbol}")
            return None
        
        # Load sentiment data
        sentiment_df = self.load_sentiment_data(symbol)
        
        # Merge price and sentiment
        df = self.merge_price_sentiment(price_df, sentiment_df, strategy=sentiment_strategy)
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Calculate sentiment features
        df = self.calculate_sentiment_features(df)
        
        # Create supervised dataset
        result = self.create_supervised_dataset(df, lookback=lookback, 
                                                predict_change=predict_change)
        
        if result[0] is None:
            self.logger.error(f"Failed to create supervised dataset for {symbol}")
            return None
        
        X, y, dates, current_prices = result
        
        # Train-test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        test_dates = dates[split_idx:]
        test_current_prices = current_prices[split_idx:]
        
        # Scale features (fit only on training data)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        sentiment_available = sentiment_df is not None and not sentiment_df.empty
        
        self.logger.info(f"{symbol}: Training={len(X_train)}, Test={len(X_test)}, "
                        f"Sentiment={'Yes' if sentiment_available else 'No (using neutral)'}")
        
        return {
            'symbol': symbol,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'test_dates': test_dates,
            'test_current_prices': test_current_prices,
            'feature_count': X_train.shape[1],
            'predict_change': predict_change,
            'has_sentiment': sentiment_available,
            'sentiment_strategy': sentiment_strategy
        }
    
    def get_all_symbols(self):
        """Get all active stock symbols from database (top 20 only)"""
        conn = self.connect_db()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM stocks WHERE is_active = 1 ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return symbols


# ========================================
# CELL 3: Initialize Feature Engineering
# ========================================

efe = EnhancedFeatureEngineering(db_config)
print("✓ Enhanced Feature Engineering initialized")
print("✓ Sentiment strategy options:")
print("  - forward_fill (RECOMMENDED): Last sentiment persists")
print("  - drop_missing: Only days with real sentiment")
print("  - missing_indicator: Use -999 for missing")
print("  - neutral_zero: Old method (fills with 0)")


# ========================================
# CELL 4: Test on Single Stock (AAPL)
# ========================================

print("\nTesting enhanced feature engineering on AAPL...")
print("="*60)

# Test with forward_fill strategy (RECOMMENDED)
test_data = efe.prepare_training_data(
    'AAPL', 
    lookback=60, 
    test_size=0.2, 
    predict_change=True,
    sentiment_strategy='forward_fill'
)

if test_data:
    print(f"\nSymbol: {test_data['symbol']}")
    print(f"Training samples: {len(test_data['X_train'])}")
    print(f"Test samples: {len(test_data['X_test'])}")
    print(f"Features per sample: {test_data['feature_count']}")
    print(f"Sentiment data available: {'Yes' if test_data['has_sentiment'] else 'No'}")
    print(f"Sentiment strategy: {test_data['sentiment_strategy']}")
    print(f"Target variable: {'Price Change %' if test_data['predict_change'] else 'Absolute Price'}")
    print(f"Target range: {test_data['y_train'].min():.2f}% to {test_data['y_train'].max():.2f}%")
    print(f"Test date range: {test_data['test_dates'][0]} to {test_data['test_dates'][-1]}")
    print("\n✓ Enhanced feature engineering working correctly!")
else:
    print("✗ Feature engineering failed")


# ========================================
# CELL 5: Compare Sentiment Strategies (Optional)
# ========================================

print("\n" + "="*60)
print("COMPARING SENTIMENT STRATEGIES ON AAPL")
print("="*60)

strategies = ['forward_fill', 'drop_missing', 'missing_indicator', 'neutral_zero']
strategy_results = {}

for strategy in strategies:
    print(f"\nTesting strategy: {strategy}...")
    try:
        data = efe.prepare_training_data(
            'AAPL',
            lookback=60,
            test_size=0.2,
            predict_change=True,
            sentiment_strategy=strategy
        )
        if data:
            strategy_results[strategy] = {
                'train_samples': len(data['X_train']),
                'test_samples': len(data['X_test']),
                'features': data['feature_count']
            }
            print(f"  Train: {len(data['X_train'])}, Test: {len(data['X_test'])}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*60)
print("STRATEGY COMPARISON SUMMARY")
print("="*60)
for strategy, results in strategy_results.items():
    print(f"{strategy:20s} | Train: {results['train_samples']:4d} | Test: {results['test_samples']:3d}")

print("\n⚠️  RECOMMENDATION: Use 'forward_fill' for best balance")
print("   - Preserves all price data")
print("   - Assumes sentiment persists (reasonable for stocks)")
print("   - Avoids fake neutral signals")


# ========================================
# CELL 6: Prepare Data for All Top 20 Stocks
# ========================================

print("\n" + "="*60)
print("Preparing enhanced training data for all TOP 20 stocks...")
print("(Price + Sentiment + Time Features + % Change Target)")
print("This will take 3-4 minutes...")
print("="*60 + "\n")

# Get active stocks only (top 20)
symbols = efe.get_all_symbols()
print(f"Found {len(symbols)} active stocks to process")
print(f"Stocks: {symbols}\n")

enhanced_stock_data = {}
failed_stocks = []
sentiment_counts = {'with_sentiment': 0, 'without_sentiment': 0}

for idx, symbol in enumerate(symbols, 1):
    print(f"[{idx}/{len(symbols)}] Processing {symbol}...", end='')
    
    try:
        data = efe.prepare_training_data(
            symbol, 
            lookback=60, 
            test_size=0.2, 
            predict_change=True,
            sentiment_strategy='forward_fill'  # RECOMMENDED STRATEGY
        )
        
        if data and len(data['X_train']) > 0:
            enhanced_stock_data[symbol] = data
            if data['has_sentiment']:
                sentiment_counts['with_sentiment'] += 1
                print(f" ✓ {len(data['X_train'])} samples [+SENTIMENT]")
            else:
                sentiment_counts['without_sentiment'] += 1
                print(f" ✓ {len(data['X_train'])} samples [neutral sentiment]")
        else:
            failed_stocks.append(symbol)
            print(f" ✗ Failed")
    except Exception as e:
        failed_stocks.append(symbol)
        print(f" ✗ Error: {str(e)[:50]}")

print("\n" + "="*60)
print("SUMMARY:")
print(f"Successfully processed: {len(enhanced_stock_data)} stocks")
print(f"  - With sentiment data: {sentiment_counts['with_sentiment']} stocks")
print(f"  - Without sentiment (neutral): {sentiment_counts['without_sentiment']} stocks")
print(f"Failed: {len(failed_stocks)} stocks")
if failed_stocks:
    print(f"Failed stocks: {', '.join(failed_stocks)}")


# ========================================
# CELL 7: Save Prepared Data
# ========================================

print("\n" + "="*60)
print("Saving prepared data...")

with open('enhanced_stock_data.pkl', 'wb') as f:
    pickle.dump(enhanced_stock_data, f)

print(f"✓ Saved data for {len(enhanced_stock_data)} stocks to enhanced_stock_data.pkl")
print("\n✓ All enhanced training data prepared!")
print("✓ Target: Price Change % (better for directional accuracy)")
print("✓ Features: Price + Sentiment + Time")
print("✓ Sentiment strategy: forward_fill (last known value persists)")
print("\n📊 Ready for enhanced model training with hyperparameter tuning!")


# ========================================
# CELL 8: Feature Analysis (Optional)
# ========================================

print("\n" + "="*60)
print("FEATURE ANALYSIS")
print("="*60)

if enhanced_stock_data:
    # Sample one stock for analysis
    sample_symbol = list(enhanced_stock_data.keys())[0]
    sample_data = enhanced_stock_data[sample_symbol]
    
    print(f"\nAnalyzing features from {sample_symbol}:")
    print(f"Total features: {sample_data['feature_count']}")
    print(f"Lookback window: 60 days")
    print(f"Features per day: {sample_data['feature_count'] // 60}")
    
    print("\nFeature categories:")
    print("  - Price indicators: ~38 features")
    print("  - Sentiment indicators: ~15 features")
    print("  - Time-based: ~4 features")
    
    print(f"\nTarget variable (y) statistics for {sample_symbol}:")
    print(f"  Mean change: {sample_data['y_train'].mean():.3f}%")
    print(f"  Std dev: {sample_data['y_train'].std():.3f}%")
    print(f"  Min: {sample_data['y_train'].min():.3f}%")
    print(f"  Max: {sample_data['y_train'].max():.3f}%")
    print(f"  Positive days: {(sample_data['y_train'] > 0).sum()} ({(sample_data['y_train'] > 0).mean()*100:.1f}%)")
    print(f"  Negative days: {(sample_data['y_train'] < 0).sum()} ({(sample_data['y_train'] < 0).mean()*100:.1f}%)")
else:
    print("No data available for analysis")
