import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self, db_config):
        self.db_config = db_config
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('feature_engineering.log'),
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
        
        # Price momentum
        df['momentum_1'] = df['close_price'].pct_change(1)
        df['momentum_5'] = df['close_price'].pct_change(5)
        df['momentum_10'] = df['close_price'].pct_change(10)
        
        # Volatility
        df['volatility_5'] = df['close_price'].rolling(window=5).std()
        df['volatility_20'] = df['close_price'].rolling(window=20).std()
        
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
        
        return df
    
    def create_supervised_dataset(self, df, lookback=60, forecast_horizon=1):
        """
        Create supervised learning dataset
        Use past 'lookback' days to predict 'forecast_horizon' days ahead
        """
        if df is None or len(df) < lookback + forecast_horizon:
            self.logger.warning("Insufficient data for supervised dataset creation")
            return None, None
        
        # Drop rows with NaN values from technical indicators
        df = df.dropna()
        
        if len(df) < lookback + forecast_horizon:
            self.logger.warning("Insufficient data after dropping NaN values")
            return None, None
        
        feature_columns = [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_diff',
            'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width',
            'momentum_1', 'momentum_5', 'momentum_10',
            'volatility_5', 'volatility_20',
            'volume_MA_5', 'volume_MA_20', 'volume_ratio',
            'daily_range', 'daily_return', 'ATR', 'OBV'
        ]
        
        X_list = []
        y_list = []
        dates_list = []
        
        for i in range(lookback, len(df) - forecast_horizon + 1):
            # Features: past 'lookback' days
            X_window = df[feature_columns].iloc[i-lookback:i].values
            
            # Flatten the window into a single feature vector
            X_flat = X_window.flatten()
            
            # Target: closing price 'forecast_horizon' days ahead
            y_target = df['close_price'].iloc[i + forecast_horizon - 1]
            
            X_list.append(X_flat)
            y_list.append(y_target)
            dates_list.append(df.index[i + forecast_horizon - 1])
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y, dates_list
    
    def prepare_training_data(self, symbol, lookback=60, test_size=0.2):
        """
        Complete pipeline: load data, calculate indicators, create supervised dataset
        Returns: X_train, X_test, y_train, y_test, scaler, test_dates
        """
        self.logger.info(f"Preparing training data for {symbol}")
        
        # Load price data
        df = self.load_price_data(symbol)
        
        if df is None or df.empty:
            self.logger.error(f"No data available for {symbol}")
            return None
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Create supervised dataset
        X, y, dates = self.create_supervised_dataset(df, lookback=lookback)
        
        if X is None:
            self.logger.error(f"Failed to create supervised dataset for {symbol}")
            return None
        
        # Train-test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        test_dates = dates[split_idx:]
        
        # Scale features (fit only on training data)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.logger.info(f"{symbol}: Training samples={len(X_train)}, Test samples={len(X_test)}")
        
        return {
            'symbol': symbol,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'test_dates': test_dates,
            'feature_count': X_train.shape[1]
        }
    
    def get_all_symbols(self):
        """Get all stock symbols from database"""
        conn = self.connect_db()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM stocks ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        return symbols
    
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

# Initialize feature engineering
fe = FeatureEngineering(db_config)
print("✓ Feature Engineering initialized")

print("Testing feature engineering on AAPL...")
print("="*60)

# Prepare data for one stock
test_data = fe.prepare_training_data('AAPL', lookback=60, test_size=0.2)

if test_data:
    print(f"Symbol: {test_data['symbol']}")
    print(f"Training samples: {len(test_data['X_train'])}")
    print(f"Test samples: {len(test_data['X_test'])}")
    print(f"Features per sample: {test_data['feature_count']}")
    print(f"Target variable (y) range: ${test_data['y_train'].min():.2f} - ${test_data['y_train'].max():.2f}")
    print(f"Test date range: {test_data['test_dates'][0]} to {test_data['test_dates'][-1]}")
    print("\n✓ Feature engineering working correctly!")
else:
    print("✗ Feature engineering failed")
    
print("Preparing training data for all stocks...")

symbols = fe.get_all_symbols()
print(f"Found {len(symbols)} stocks to process")

all_stock_data = {}
failed_stocks = []

for idx, symbol in enumerate(symbols, 1):
    print(f"[{idx}/{len(symbols)}] Processing {symbol}...", end='')
    
    try:
        data = fe.prepare_training_data(symbol, lookback=60, test_size=0.2)
        
        if data and len(data['X_train']) > 0:
            all_stock_data[symbol] = data
            print(f" ✓ {len(data['X_train'])} train samples")
        else:
            failed_stocks.append(symbol)
            print(f" ✗ Failed")
    except Exception as e:
        failed_stocks.append(symbol)
        print(f" ✗ Error: {str(e)[:50]}")

print("\n" + "="*60)
print("SUMMARY:")
print(f"Successfully processed: {len(all_stock_data)} stocks")
print(f"Failed: {len(failed_stocks)} stocks")
if failed_stocks:
    print(f"Failed stocks: {', '.join(failed_stocks)}")

print("\n✓ All training data prepared!")
print("Ready for model training in next step.")

# Save prepared data
import pickle

with open('prepared_stock_data.pkl', 'wb') as f:
    pickle.dump(all_stock_data, f)

print(f"✓ Saved data for {len(all_stock_data)} stocks to prepared_stock_data.pkl")