# ========================================
# CELL 1: Imports and LSTM Model Class
# ========================================
import numpy as np
import pandas as pd
import mysql.connector
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class LSTMStockPredictor:
    def __init__(self, db_config, model_save_dir='models/lstm_models'):
        self.db_config = db_config
        self.model_save_dir = model_save_dir
        self.setup_logging()
        self.create_model_directory()
        
    def setup_logging(self):
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/lstm_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_model_directory(self):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            self.logger.info(f"Created LSTM model directory: {self.model_save_dir}")
    
    def connect_db(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except mysql.connector.Error as e:
            self.logger.error(f"Database connection error: {e}")
            return None
    
    def prepare_lstm_data(self, X_train, X_test, lookback, n_features):
        """
        Reshape data for LSTM input
        LSTM expects: (samples, timesteps, features)
        """
        # Reshape from (samples, lookback*features) to (samples, lookback, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], lookback, n_features))
        X_test_lstm = X_test.reshape((X_test.shape[0], lookback, n_features))
        
        return X_train_lstm, X_test_lstm
    
    def build_lstm_model(self, lookback, n_features, architecture='standard'):
        """
        Build LSTM model with different architectures
        """
        model = Sequential()
        
        if architecture == 'simple':
            # Simple LSTM
            model.add(LSTM(50, input_shape=(lookback, n_features)))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
        elif architecture == 'standard':
            # Standard stacked LSTM
            model.add(LSTM(100, return_sequences=True, input_shape=(lookback, n_features)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(25))
            model.add(Dense(1))
            
        elif architecture == 'deep':
            # Deep LSTM with batch normalization
            model.add(LSTM(128, return_sequences=True, input_shape=(lookback, n_features)))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(LSTM(64, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(16))
            model.add(Dense(1))
            
        elif architecture == 'bidirectional':
            # Bidirectional LSTM
            model.add(Bidirectional(LSTM(64, return_sequences=True), 
                                   input_shape=(lookback, n_features)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(32, return_sequences=False)))
            model.add(Dropout(0.2))
            model.add(Dense(16))
            model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, 
                   lookback=60, architecture='standard', epochs=100, batch_size=32):
        """
        Train LSTM model
        """
        self.logger.info(f"Training LSTM ({architecture} architecture)...")
        
        # Calculate number of features
        n_features = X_train.shape[1] // lookback
        
        # Reshape data for LSTM
        X_train_lstm, X_test_lstm = self.prepare_lstm_data(X_train, X_test, lookback, n_features)
        
        # Build model
        model = self.build_lstm_model(lookback, n_features, architecture)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train_lstm, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test_lstm, verbose=0).flatten()
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.logger.info(f"LSTM ({architecture}) - Dir Acc: {metrics['directional_accuracy']:.2f}%, "
                        f"RMSE: {metrics['rmse']:.4f}")
        
        return model, y_pred, metrics, history
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = -999
        
        # Directional accuracy
        true_direction = y_true > 0
        pred_direction = y_pred > 0
        dir_acc = (np.sum(true_direction == pred_direction) / len(true_direction)) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'directional_accuracy': dir_acc
        }
    
    def save_model(self, model, symbol, architecture):
        """Save LSTM model"""
        filename = f"{symbol}_LSTM_{architecture}.h5"
        filepath = os.path.join(self.model_save_dir, filename)
        model.save(filepath)
        return filepath
    
    def save_performance_to_db(self, symbol, architecture, metrics, train_samples, 
                               test_samples, feature_count, model_path, has_sentiment):
        """Save model performance to database"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        model_type_label = f"LSTM_{architecture}_{'Sentiment' if has_sentiment else 'NoSentiment'}"
        
        query = """
        INSERT INTO model_performance 
        (symbol, model_type, rmse, mae, r2_score, directional_accuracy,
         train_samples, test_samples, feature_count, model_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            symbol, model_type_label,
            metrics['rmse'], metrics['mae'], metrics['r2_score'],
            metrics['directional_accuracy'],
            train_samples, test_samples, feature_count, model_path
        )
        
        try:
            cursor.execute(query, values)
            conn.commit()
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error saving performance: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def train_all_architectures(self, stock_data, architectures=['simple', 'standard', 'bidirectional']):
        """
        Train multiple LSTM architectures for a stock
        """
        symbol = stock_data['symbol']
        X_train = stock_data['X_train']
        X_test = stock_data['X_test']
        y_train = stock_data['y_train']
        y_test = stock_data['y_test']
        has_sentiment = stock_data.get('has_sentiment', False)
        
        results = {}
        
        for arch in architectures:
            try:
                model, y_pred, metrics, history = self.train_lstm(
                    X_train, y_train, X_test, y_test,
                    lookback=60, architecture=arch, epochs=100, batch_size=32
                )
                
                if model:
                    model_path = self.save_model(model, symbol, arch)
                    self.save_performance_to_db(
                        symbol, arch, metrics,
                        len(X_train), len(X_test), X_train.shape[1], 
                        model_path, has_sentiment
                    )
                    
                    results[f'LSTM_{arch}'] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': y_pred,
                        'history': history
                    }
                    
            except Exception as e:
                self.logger.error(f"LSTM {arch} failed for {symbol}: {e}")
        
        return results


# ========================================
# CELL 2: Initialize LSTM Trainer
# ========================================
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

lstm_trainer = LSTMStockPredictor(db_config)
print("✓ LSTM Trainer initialized")
print("✓ TensorFlow version:", tf.__version__)
print("✓ GPU available:", "Yes" if tf.config.list_physical_devices('GPU') else "No (CPU mode)")


# ========================================
# CELL 3: Load Top 20 Stock Data
# ========================================
print("\nLoading top 20 stock data...")

try:
    with open('data/enhanced_stock_data_top20.pkl', 'rb') as f:
        top20_stock_data = pickle.load(f)
    print(f"✓ Loaded data for {len(top20_stock_data)} stocks")
    print(f"Stocks: {list(top20_stock_data.keys())}")
except FileNotFoundError:
    print("⚠️  Run 06_portfolio_optimizer.ipynb first to create top20 data")
    top20_stock_data = {}


# ========================================
# CELL 4: Train LSTM for One Stock (Test)
# ========================================
if top20_stock_data:
    test_symbol = list(top20_stock_data.keys())[0]
    print(f"\nTesting LSTM training on {test_symbol}...")
    print("="*70)
    print("Training 3 LSTM architectures (simple, standard, bidirectional)")
    print("This will take 5-10 minutes...\n")
    
    results = lstm_trainer.train_all_architectures(
        top20_stock_data[test_symbol],
        architectures=['simple', 'standard', 'bidirectional']
    )
    
    if results:
        print("\nLSTM Performance Comparison:")
        print("-"*70)
        print(f"{'Architecture':<20} {'RMSE (%)':<12} {'MAE (%)':<12} {'Dir Acc'}")
        print("-"*70)
        
        for model_name, data in results.items():
            metrics = data['metrics']
            print(f"{model_name:<20} {metrics['rmse']:>8.3f}%    "
                  f"{metrics['mae']:>8.3f}%    {metrics['directional_accuracy']:>8.1f}%")
        
        # Find best architecture
        best_arch = max(results.items(), 
                       key=lambda x: x[1]['metrics']['directional_accuracy'])
        print(f"\n✓ Best architecture: {best_arch[0]} "
              f"({best_arch[1]['metrics']['directional_accuracy']:.1f}% accuracy)")
    else:
        print("✗ LSTM training failed")


# ========================================
# CELL 5: Train LSTM for All Top 20 Stocks
# ========================================
if top20_stock_data:
    print("\n" + "="*70)
    print("Training LSTM models for all top 20 stocks...")
    print("Using best architecture: 'standard' (based on test)")
    print("This will take 20-40 minutes...")
    print("="*70 + "\n")
    
    all_lstm_results = {}
    lstm_summary = []
    
    for idx, (symbol, stock_data) in enumerate(top20_stock_data.items(), 1):
        print(f"\n[{idx}/{len(top20_stock_data)}] Training LSTM for {symbol}...")
        print("-"*70)
        
        try:
            # Train only standard architecture for speed
            results = lstm_trainer.train_all_architectures(
                stock_data,
                architectures=['standard']
            )
            
            if results:
                all_lstm_results[symbol] = results
                
                for model_name, data in results.items():
                    metrics = data['metrics']
                    print(f"  {model_name:<15} | RMSE: {metrics['rmse']:6.2f}% | "
                          f"Dir Acc: {metrics['directional_accuracy']:5.1f}%")
                    
                    lstm_summary.append({
                        'symbol': symbol,
                        'model': model_name,
                        'rmse': metrics['rmse'],
                        'dir_acc': metrics['directional_accuracy'],
                        'has_sentiment': stock_data.get('has_sentiment', False)
                    })
            else:
                print(f"  ✗ LSTM training failed for {symbol}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:70]}")
    
    print("\n" + "="*70)
    print("LSTM TRAINING COMPLETE!")
    print("="*70)
    
    # ========================================
    # CELL 6: LSTM Performance Summary
    # ========================================
    lstm_df = pd.DataFrame(lstm_summary)
    
    if not lstm_df.empty:
        print("\n" + "="*70)
        print("LSTM PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"\nAverage Directional Accuracy: {lstm_df['dir_acc'].mean():.2f}%")
        print(f"Average RMSE: {lstm_df['rmse'].mean():.3f}%")
        print(f"Best performing stock: {lstm_df.loc[lstm_df['dir_acc'].idxmax(), 'symbol']} "
              f"({lstm_df['dir_acc'].max():.1f}%)")
        print(f"Worst performing stock: {lstm_df.loc[lstm_df['dir_acc'].idxmin(), 'symbol']} "
              f"({lstm_df['dir_acc'].min():.1f}%)")
        
        print("\nTop 5 LSTM Predictions:")
        print("-"*70)
        top_5 = lstm_df.nlargest(5, 'dir_acc')
        for _, row in top_5.iterrows():
            print(f"{row['symbol']:6s} | Dir Acc: {row['dir_acc']:5.1f}% | RMSE: {row['rmse']:6.2f}%")
        
        # Save LSTM results
        with open('data/lstm_results.pkl', 'wb') as f:
            pickle.dump(all_lstm_results, f)
        
        print("\n✓ LSTM results saved to: data/lstm_results.pkl")
        print("✓ Ready for ensemble methods!")
    
else:
    print("No stock data loaded. Run portfolio optimizer first.")