import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
import logging
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, db_config, model_save_dir='trained_models'):
        self.db_config = db_config
        self.model_save_dir = model_save_dir
        self.setup_logging()
        self.create_model_directory()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('model_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_model_directory(self):
        """Create directory to save trained models"""
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            self.logger.info(f"Created model directory: {self.model_save_dir}")
    
    def connect_db(self):
        try:
            conn = mysql.connector.connect(**self.db_config)
            return conn
        except mysql.connector.Error as e:
            self.logger.error(f"Database connection error: {e}")
            return None
    
    def create_model_tables(self):
        """Create tables to store model performance and metadata"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Model performance tracking
        performance_table = """
        CREATE TABLE IF NOT EXISTS model_performance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            model_type VARCHAR(50),
            rmse DECIMAL(10,4),
            mae DECIMAL(10,4),
            r2_score DECIMAL(10,6),
            directional_accuracy DECIMAL(5,2),
            train_samples INT,
            test_samples INT,
            feature_count INT,
            model_path VARCHAR(255),
            trained_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_symbol (symbol),
            INDEX idx_model_type (model_type)
        )
        """
        
        # Model selection (which model to use for each stock)
        selection_table = """
        CREATE TABLE IF NOT EXISTS model_selection (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10) UNIQUE,
            selected_model_type VARCHAR(50),
            model_path VARCHAR(255),
            rmse DECIMAL(10,4),
            selection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        """
        
        try:
            cursor.execute(performance_table)
            cursor.execute(selection_table)
            conn.commit()
            self.logger.info("Model tables created successfully")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error creating model tables: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate percentage of correct directional predictions"""
        if len(y_true) <= 1:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        correct = np.sum(true_direction == pred_direction)
        accuracy = (correct / len(true_direction)) * 100
        
        return accuracy
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        self.logger.info("Training Random Forest...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        return model, y_pred, metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        self.logger.info("Training XGBoost...")
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        return model, y_pred, metrics
    
    def train_arima(self, y_train, y_test):
        """Train ARIMA model (uses only target values, not features)"""
        self.logger.info("Training ARIMA...")
        
        try:
            # ARIMA works on the time series directly
            model = ARIMA(y_train, order=(2, 1, 1))
            fitted_model = model.fit()
            
            # Forecast for test period
            forecast = fitted_model.forecast(steps=len(y_test))
            y_pred = np.array(forecast)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            
            return fitted_model, y_pred, metrics
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            return None, None, None
    
    def train_mlp(self, X_train, y_train, X_test, y_test):
        """Train MLP Neural Network"""
        self.logger.info("Training MLP...")
        
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        return model, y_pred, metrics
    
    def train_lasso(self, X_train, y_train, X_test, y_test):
        """Train Lasso Regression"""
        self.logger.info("Training Lasso...")
        
        model = Lasso(alpha=1.0, random_state=42, max_iter=5000)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        return model, y_pred, metrics
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        dir_acc = self.calculate_directional_accuracy(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'directional_accuracy': dir_acc
        }
    
    def save_model(self, model, symbol, model_type):
        """Save trained model to disk"""
        filename = f"{symbol}_{model_type}.pkl"
        filepath = os.path.join(self.model_save_dir, filename)
        joblib.dump(model, filepath)
        return filepath
    
    def save_performance_to_db(self, symbol, model_type, metrics, train_samples, 
                               test_samples, feature_count, model_path):
        """Save model performance metrics to database"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        query = """
        INSERT INTO model_performance 
        (symbol, model_type, rmse, mae, r2_score, directional_accuracy,
         train_samples, test_samples, feature_count, model_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            symbol,
            model_type,
            metrics['rmse'],
            metrics['mae'],
            metrics['r2_score'],
            metrics['directional_accuracy'],
            train_samples,
            test_samples,
            feature_count,
            model_path
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
    
    def train_all_models(self, stock_data):
        """
        Train all 5 model types for a given stock
        Returns: dict of results for each model
        """
        symbol = stock_data['symbol']
        X_train = stock_data['X_train']
        X_test = stock_data['X_test']
        y_train = stock_data['y_train']
        y_test = stock_data['y_test']
        
        results = {}
        
        # 1. Random Forest
        try:
            model, y_pred, metrics = self.train_random_forest(X_train, y_train, X_test, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'RandomForest')
                self.save_performance_to_db(
                    symbol, 'RandomForest', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path
                )
                results['RandomForest'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"Random Forest failed for {symbol}: {e}")
        
        # 2. XGBoost
        try:
            model, y_pred, metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'XGBoost')
                self.save_performance_to_db(
                    symbol, 'XGBoost', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path
                )
                results['XGBoost'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"XGBoost failed for {symbol}: {e}")
        
        # 3. ARIMA
        try:
            model, y_pred, metrics = self.train_arima(y_train, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'ARIMA')
                self.save_performance_to_db(
                    symbol, 'ARIMA', metrics,
                    len(y_train), len(y_test), 0, model_path
                )
                results['ARIMA'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"ARIMA failed for {symbol}: {e}")
        
        # 4. MLP
        try:
            model, y_pred, metrics = self.train_mlp(X_train, y_train, X_test, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'MLP')
                self.save_performance_to_db(
                    symbol, 'MLP', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path
                )
                results['MLP'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"MLP failed for {symbol}: {e}")
        
        # 5. Lasso
        try:
            model, y_pred, metrics = self.train_lasso(X_train, y_train, X_test, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'Lasso')
                self.save_performance_to_db(
                    symbol, 'Lasso', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path
                )
                results['Lasso'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"Lasso failed for {symbol}: {e}")
        
        return results
    
    def select_best_model(self, symbol, results):
        """Select best model based on RMSE and save to database"""
        if not results:
            self.logger.warning(f"No valid models for {symbol}")
            return None
        
        best_model_type = None
        best_rmse = float('inf')
        best_model_path = None
        
        for model_type, data in results.items():
            rmse = data['metrics']['rmse']
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_type = model_type
                best_model_path = f"{self.model_save_dir}/{symbol}_{model_type}.pkl"
        
        # Save selection to database
        conn = self.connect_db()
        if not conn:
            return None
        
        cursor = conn.cursor()
        
        query = """
        INSERT INTO model_selection (symbol, selected_model_type, model_path, rmse, notes)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            selected_model_type = VALUES(selected_model_type),
            model_path = VALUES(model_path),
            rmse = VALUES(rmse),
            selection_date = CURRENT_TIMESTAMP,
            notes = VALUES(notes)
        """
        
        notes = f"Best of {len(results)} models. Dir Acc: {results[best_model_type]['metrics']['directional_accuracy']:.1f}%"
        
        try:
            cursor.execute(query, (symbol, best_model_type, best_model_path, best_rmse, notes))
            conn.commit()
            self.logger.info(f"{symbol}: Selected {best_model_type} (RMSE={best_rmse:.2f})")
        except mysql.connector.Error as e:
            self.logger.error(f"Error saving model selection: {e}")
        finally:
            cursor.close()
            conn.close()
        
        return best_model_type
    
    db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'trading_system'
}

trainer = ModelTrainer(db_config)
trainer.create_model_tables()
print("✓ Model Trainer initialized")
print("✓ Model performance tables created")

print("Testing model training on AAPL...")
print("="*60)

# Load prepared data
import pickle

with open('prepared_stock_data.pkl', 'rb') as f:
    all_stock_data = pickle.load(f)

print(f"✓ Loaded data for {len(all_stock_data)} stocks")

# Use the prepared data from feature engineering
test_symbol = 'AAPL'
if test_symbol in all_stock_data:
    print(f"Training all 5 models for {test_symbol}...")
    results = trainer.train_all_models(all_stock_data[test_symbol])
    
    print("\nModel Performance:")
    print("-"*60)
    for model_type, data in results.items():
        metrics = data['metrics']
        print(f"{model_type:15s} | RMSE: ${metrics['rmse']:8.2f} | "
              f"MAE: ${metrics['mae']:8.2f} | "
              f"Dir Acc: {metrics['directional_accuracy']:5.1f}%")
    
    # Select best model
    best = trainer.select_best_model(test_symbol, results)
    print(f"\n✓ Best model for {test_symbol}: {best}")
else:
    print(f"No data available for {test_symbol}")
    
print("\n" + "="*60)
print("Training models for all stocks...")
print("This will take 10-20 minutes depending on your CPU")
print("="*60 + "\n")

all_results = {}
training_summary = []

for idx, (symbol, stock_data) in enumerate(all_stock_data.items(), 1):
    print(f"\n[{idx}/{len(all_stock_data)}] Training models for {symbol}...")
    print("-"*60)
    
    try:
        results = trainer.train_all_models(stock_data)
        
        if results:
            all_results[symbol] = results
            
            # Print performance
            for model_type, data in results.items():
                metrics = data['metrics']
                print(f"  {model_type:12s} | RMSE: ${metrics['rmse']:7.2f} | Dir Acc: {metrics['directional_accuracy']:5.1f}%")
                
                training_summary.append({
                    'symbol': symbol,
                    'model': model_type,
                    'rmse': metrics['rmse'],
                    'dir_acc': metrics['directional_accuracy']
                })
            
            # Select best model
            best = trainer.select_best_model(symbol, results)
            print(f"  → Best: {best}")
        else:
            print(f"  ✗ No models succeeded for {symbol}")
            
    except Exception as e:
        print(f"  ✗ Training failed: {str(e)[:50]}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

import pandas as pd

# Create summary dataframe
summary_df = pd.DataFrame(training_summary)

if not summary_df.empty:
    print("\n" + "="*60)
    print("OVERALL TRAINING SUMMARY")
    print("="*60)
    
    # Best performers by model type
    print("\nBest RMSE by Model Type:")
    print("-"*60)
    for model_type in summary_df['model'].unique():
        model_data = summary_df[summary_df['model'] == model_type]
        best_idx = model_data['rmse'].idxmin()
        best_row = model_data.loc[best_idx]
        print(f"{model_type:15s} | {best_row['symbol']:6s} | RMSE: ${best_row['rmse']:8.2f}")
    
    # Average performance
    print("\nAverage Performance by Model Type:")
    print("-"*60)
    avg_performance = summary_df.groupby('model').agg({
        'rmse': 'mean',
        'dir_acc': 'mean'
    }).round(2)
    print(avg_performance)
    
    # Best stocks (lowest RMSE across all models)
    print("\nTop 10 Stocks by Average RMSE:")
    print("-"*60)
    stock_avg = summary_df.groupby('symbol')['rmse'].mean().sort_values().head(10)
    for symbol, rmse in stock_avg.items():
        print(f"{symbol:6s} | Avg RMSE: ${rmse:8.2f}")
    
    # Check model selection table
    conn = mysql.connector.connect(**db_config)
    selection_df = pd.read_sql("""
        SELECT symbol, selected_model_type, rmse, notes
        FROM model_selection
        ORDER BY rmse ASC
    """, conn)
    conn.close()
    
    print("\nSelected Models for Each Stock:")
    print("-"*60)
    model_counts = selection_df['selected_model_type'].value_counts()
    for model, count in model_counts.items():
        print(f"{model:15s}: {count} stocks")
    
    print("\n✓ All models trained and evaluated!")
    print(f"✓ Model files saved in: {trainer.model_save_dir}/")
    print("✓ Performance metrics saved to database")
    print("\nReady for Week 4: Trading System Implementation")
else:
    print("No training summary available")
    
