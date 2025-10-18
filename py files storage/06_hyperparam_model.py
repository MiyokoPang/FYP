import numpy as np
import pandas as pd
import mysql.connector
from datetime import datetime
import logging
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelTrainer:
    def __init__(self, db_config, model_save_dir='enhanced_models'):
        self.db_config = db_config
        self.model_save_dir = model_save_dir
        self.setup_logging()
        self.create_model_directory()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_model_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_model_directory(self):
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
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """Calculate percentage of correct directional predictions"""
        # For % change predictions, direction is simply the sign
        true_direction = y_true > 0
        pred_direction = y_pred > 0
        
        correct = np.sum(true_direction == pred_direction)
        accuracy = (correct / len(true_direction)) * 100
        
        return accuracy
    
    def calculate_metrics(self, y_true, y_pred, is_percentage=True):
        """Calculate performance metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # R² can be negative
        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = -999
        
        dir_acc = self.calculate_directional_accuracy(y_true, y_pred)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'directional_accuracy': dir_acc
        }
    
    def train_random_forest_tuned(self, X_train, y_train, X_test, y_test, quick_tune=False):
        """Train Random Forest with hyperparameter tuning"""
        self.logger.info("Training Random Forest with tuning...")
        
        if quick_tune:
            # Quick tuning for faster iteration
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [15, 25],
                'min_samples_split': [3, 5],
                'min_samples_leaf': [1, 2]
            }
        else:
            # Comprehensive tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Use RandomizedSearchCV for speed
        grid_search = RandomizedSearchCV(
            rf, param_grid, n_iter=10, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.logger.info(f"Best RF params: {grid_search.best_params_}")
        
        return best_model, y_pred, metrics
    
    def train_xgboost_tuned(self, X_train, y_train, X_test, y_test, quick_tune=False):
        """Train XGBoost with hyperparameter tuning"""
        self.logger.info("Training XGBoost with tuning...")
        
        if quick_tune:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 10],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 7, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        grid_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=10, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.logger.info(f"Best XGB params: {grid_search.best_params_}")
        
        return best_model, y_pred, metrics
    
    def train_mlp_tuned(self, X_train, y_train, X_test, y_test, quick_tune=False):
        """Train MLP with hyperparameter tuning"""
        self.logger.info("Training MLP with tuning...")
        
        if quick_tune:
            param_grid = {
                'hidden_layer_sizes': [(100,), (100, 50)],
                'alpha': [0.0001, 0.001],
                'learning_rate_init': [0.001]
            }
        else:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100), (150, 75)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        
        mlp = MLPRegressor(
            max_iter=1000, early_stopping=True, 
            validation_fraction=0.1, random_state=42
        )
        
        grid_search = RandomizedSearchCV(
            mlp, param_grid, n_iter=8, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.logger.info(f"Best MLP params: {grid_search.best_params_}")
        
        return best_model, y_pred, metrics
    
    def train_lasso_tuned(self, X_train, y_train, X_test, y_test):
        """Train Lasso with hyperparameter tuning"""
        self.logger.info("Training Lasso with tuning...")
        
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
        
        lasso = Lasso(max_iter=5000, random_state=42)
        
        grid_search = GridSearchCV(
            lasso, param_grid, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.logger.info(f"Best Lasso alpha: {grid_search.best_params_['alpha']}")
        
        return best_model, y_pred, metrics
    
    def train_ridge_tuned(self, X_train, y_train, X_test, y_test):
        """Train Ridge Regression with hyperparameter tuning"""
        self.logger.info("Training Ridge with tuning...")
        
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
        
        ridge = Ridge(max_iter=5000, random_state=42)
        
        grid_search = GridSearchCV(
            ridge, param_grid, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        metrics = self.calculate_metrics(y_test, y_pred)
        
        self.logger.info(f"Best Ridge alpha: {grid_search.best_params_['alpha']}")
        
        return best_model, y_pred, metrics
    
    def train_arima(self, y_train, y_test):
        """Train ARIMA model (no hyperparameter tuning for now)"""
        self.logger.info("Training ARIMA...")
        
        try:
            model = ARIMA(y_train, order=(2, 1, 1))
            fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps=len(y_test))
            y_pred = np.array(forecast)
            
            metrics = self.calculate_metrics(y_test, y_pred)
            
            return fitted_model, y_pred, metrics
        except Exception as e:
            self.logger.error(f"ARIMA training failed: {e}")
            return None, None, None
    
    def save_model(self, model, symbol, model_type):
        """Save trained model to disk"""
        filename = f"{symbol}_{model_type}_enhanced.pkl"
        filepath = os.path.join(self.model_save_dir, filename)
        joblib.dump(model, filepath)
        return filepath
    
    def save_performance_to_db(self, symbol, model_type, metrics, train_samples, 
                               test_samples, feature_count, model_path, has_sentiment):
        """Save model performance metrics to database"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Use existing model_performance table
        query = """
        INSERT INTO model_performance 
        (symbol, model_type, rmse, mae, r2_score, directional_accuracy,
         train_samples, test_samples, feature_count, model_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        model_type_label = f"{model_type}_Enhanced_{'Sentiment' if has_sentiment else 'NoSentiment'}"
        
        values = (
            symbol,
            model_type_label,
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
    
    def train_all_models(self, stock_data, quick_tune=True):
        """
        Train all models with hyperparameter tuning
        quick_tune: If True, use faster parameter grids
        """
        symbol = stock_data['symbol']
        X_train = stock_data['X_train']
        X_test = stock_data['X_test']
        y_train = stock_data['y_train']
        y_test = stock_data['y_test']
        has_sentiment = stock_data.get('has_sentiment', False)
        
        results = {}
        
        # 1. Random Forest (Tuned)
        try:
            model, y_pred, metrics = self.train_random_forest_tuned(
                X_train, y_train, X_test, y_test, quick_tune
            )
            if model:
                model_path = self.save_model(model, symbol, 'RandomForest')
                self.save_performance_to_db(
                    symbol, 'RandomForest', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path, has_sentiment
                )
                results['RandomForest'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"Random Forest failed for {symbol}: {e}")
        
        # 2. XGBoost (Tuned)
        try:
            model, y_pred, metrics = self.train_xgboost_tuned(
                X_train, y_train, X_test, y_test, quick_tune
            )
            if model:
                model_path = self.save_model(model, symbol, 'XGBoost')
                self.save_performance_to_db(
                    symbol, 'XGBoost', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path, has_sentiment
                )
                results['XGBoost'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"XGBoost failed for {symbol}: {e}")
        
        # 3. MLP (Tuned)
        try:
            model, y_pred, metrics = self.train_mlp_tuned(
                X_train, y_train, X_test, y_test, quick_tune
            )
            if model:
                model_path = self.save_model(model, symbol, 'MLP')
                self.save_performance_to_db(
                    symbol, 'MLP', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path, has_sentiment
                )
                results['MLP'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"MLP failed for {symbol}: {e}")
        
        # 4. Lasso (Tuned)
        try:
            model, y_pred, metrics = self.train_lasso_tuned(X_train, y_train, X_test, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'Lasso')
                self.save_performance_to_db(
                    symbol, 'Lasso', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path, has_sentiment
                )
                results['Lasso'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"Lasso failed for {symbol}: {e}")
        
        # 5. Ridge (Tuned) 
        try:
            model, y_pred, metrics = self.train_ridge_tuned(X_train, y_train, X_test, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'Ridge')
                self.save_performance_to_db(
                    symbol, 'Ridge', metrics,
                    len(X_train), len(X_test), X_train.shape[1], model_path, has_sentiment
                )
                results['Ridge'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"Ridge failed for {symbol}: {e}")
        
        # 6. ARIMA
        try:
            model, y_pred, metrics = self.train_arima(y_train, y_test)
            if model:
                model_path = self.save_model(model, symbol, 'ARIMA')
                self.save_performance_to_db(
                    symbol, 'ARIMA', metrics,
                    len(y_train), len(y_test), 0, model_path, has_sentiment
                )
                results['ARIMA'] = {'model': model, 'metrics': metrics, 'predictions': y_pred}
        except Exception as e:
            self.logger.error(f"ARIMA failed for {symbol}: {e}")
        
        return results
    
    def select_best_model(self, symbol, results):
        """Select best model based on directional accuracy (primary) and RMSE (secondary)"""
        if not results:
            self.logger.warning(f"No valid models for {symbol}")
            return None
        
        # Sort by directional accuracy first, then RMSE
        sorted_models = sorted(
            results.items(),
            key=lambda x: (-x[1]['metrics']['directional_accuracy'], x[1]['metrics']['rmse'])
        )
        
        best_model_type = sorted_models[0][0]
        best_metrics = sorted_models[0][1]['metrics']
        best_model_path = f"{self.model_save_dir}/{symbol}_{best_model_type}_enhanced.pkl"
        
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
        
        notes = f"Enhanced model. Dir Acc: {best_metrics['directional_accuracy']:.1f}%, RMSE: {best_metrics['rmse']:.2f}%"
        
        try:
            cursor.execute(query, (symbol, best_model_type + '_Enhanced', best_model_path, 
                                  best_metrics['rmse'], notes))
            conn.commit()
            self.logger.info(f"{symbol}: Selected {best_model_type} "
                           f"(Dir Acc={best_metrics['directional_accuracy']:.1f}%, "
                           f"RMSE={best_metrics['rmse']:.2f}%)")
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

enhanced_trainer = EnhancedModelTrainer(db_config)
print("✓ Enhanced Model Trainer initialized")
print("✓ Will train with hyperparameter tuning")
print("✓ Target: Price Change % (better for directional accuracy)")

# Load prepared data
import pickle

with open('enhanced_stock_data.pkl', 'rb') as f:
    enhanced_stock_data = pickle.load(f)

print(f"✓ Loaded data for {len(enhanced_stock_data)} stocks")

print("\nTesting enhanced model training on AAPL...")
print("="*60)

test_symbol = 'AAPL'
if test_symbol in enhanced_stock_data:
    print(f"Training 6 tuned models for {test_symbol}...")
    print("This will take 5-10 minutes due to hyperparameter tuning...\n")
    
    results = enhanced_trainer.train_all_models(enhanced_stock_data[test_symbol], quick_tune=True)
    
    print("\nEnhanced Model Performance:")
    print("-"*70)
    print(f"{'Model':<15} {'RMSE (%)':<12} {'MAE (%)':<12} {'Dir Acc':<12}")
    print("-"*70)
    
    for model_type, data in results.items():
        metrics = data['metrics']
        print(f"{model_type:<15} {metrics['rmse']:>8.3f}%    "
              f"{metrics['mae']:>8.3f}%    "
              f"{metrics['directional_accuracy']:>8.1f}%")
    
    best = enhanced_trainer.select_best_model(test_symbol, results)
    print(f"\n✓ Best model for {test_symbol}: {best}")
    print(f"✓ Directional Accuracy: {results[best]['metrics']['directional_accuracy']:.1f}%")
else:
    print(f"No data available for {test_symbol}")

print("\n" + "="*70)
print("Training enhanced models for all stocks with hyperparameter tuning...")
print("This will take 30-60 minutes depending on your CPU")
print("="*70 + "\n")

enhanced_results = {}
enhanced_summary = []

for idx, (symbol, stock_data) in enumerate(enhanced_stock_data.items(), 1):
    print(f"\n[{idx}/{len(enhanced_stock_data)}] Training models for {symbol}...")
    print("-"*70)
    
    try:
        results = enhanced_trainer.train_all_models(stock_data, quick_tune=True)
        
        if results:
            enhanced_results[symbol] = results
            
            # Print performance
            for model_type, data in results.items():
                metrics = data['metrics']
                print(f"  {model_type:12s} | RMSE: {metrics['rmse']:6.2f}% | "
                      f"Dir Acc: {metrics['directional_accuracy']:5.1f}%")
                
                enhanced_summary.append({
                    'symbol': symbol,
                    'model': model_type,
                    'rmse': metrics['rmse'],
                    'dir_acc': metrics['directional_accuracy'],
                    'has_sentiment': stock_data.get('has_sentiment', False)
                })
            
            # Select best model
            best = enhanced_trainer.select_best_model(symbol, results)
            best_metrics = results[best]['metrics']
            print(f"  → Best: {best} (Dir Acc: {best_metrics['directional_accuracy']:.1f}%)")
        else:
            print(f"  ✗ No models succeeded for {symbol}")
            
    except Exception as e:
        print(f"  ✗ Training failed: {str(e)[:50]}")

print("\n" + "="*70)
print("ENHANCED TRAINING COMPLETE!")
print("="*70)

import pandas as pd

summary_df = pd.DataFrame(enhanced_summary)

if not summary_df.empty:
    print("\n" + "="*70)
    print("ENHANCED MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    # Average directional accuracy by model type
    print("\nDirectional Accuracy by Model Type:")
    print("-"*70)
    dir_acc_avg = summary_df.groupby('model')['dir_acc'].agg(['mean', 'std', 'min', 'max'])
    print(dir_acc_avg.round(2))
    
    # Average RMSE by model type
    print("\nRMSE (%) by Model Type:")
    print("-"*70)
    rmse_avg = summary_df.groupby('model')['rmse'].agg(['mean', 'std', 'min', 'max'])
    print(rmse_avg.round(3))
    
    # Best performers
    print("\nTop 5 Stocks by Directional Accuracy (Best Model):")
    print("-"*70)
    best_per_stock = summary_df.loc[summary_df.groupby('symbol')['dir_acc'].idxmax()]
    top_stocks = best_per_stock.nlargest(5, 'dir_acc')[['symbol', 'model', 'dir_acc', 'rmse']]
    for _, row in top_stocks.iterrows():
        print(f"{row['symbol']:6s} | {row['model']:12s} | "
              f"Dir Acc: {row['dir_acc']:5.1f}% | RMSE: {row['rmse']:6.2f}%")
    
    # Sentiment impact
    print("\nSentiment Impact:")
    print("-"*70)
    sentiment_comparison = summary_df.groupby('has_sentiment').agg({
        'dir_acc': 'mean',
        'rmse': 'mean'
    }).round(2)
    
    # Safely rename index based on actual values
    if len(sentiment_comparison) == 2:
        sentiment_comparison.index = ['No Sentiment', 'With Sentiment']
    elif len(sentiment_comparison) == 1:
        has_sent = sentiment_comparison.index[0]
        sentiment_comparison.index = ['With Sentiment' if has_sent else 'No Sentiment (All stocks)']
    
    print(sentiment_comparison)
    
    # Show distribution
    sent_counts = summary_df.groupby('has_sentiment')['symbol'].nunique()
    print(f"\nStocks with sentiment data: {sent_counts.get(True, 0)}")
    print(f"Stocks without sentiment data: {sent_counts.get(False, 0)}")
    
    # Model selection distribution
    conn = mysql.connector.connect(**db_config)
    selection_df = pd.read_sql("""
        SELECT symbol, selected_model_type, rmse, notes
        FROM model_selection
        WHERE selected_model_type LIKE '%Enhanced%'
        ORDER BY rmse ASC
    """, conn)
    conn.close()
    
    if not selection_df.empty:
        print("\nEnhanced Model Selection:")
        print("-"*70)
        model_counts = selection_df['selected_model_type'].value_counts()
        for model, count in model_counts.items():
            print(f"{model:30s}: {count} stocks")
    
    # Overall improvement
    print("\n" + "="*70)
    print("KEY IMPROVEMENTS:")
    print("="*70)
    avg_dir_acc = summary_df['dir_acc'].mean()
    avg_rmse = summary_df['rmse'].mean()
    print(f"Average Directional Accuracy: {avg_dir_acc:.1f}% (Target: >55%)")
    print(f"Average RMSE: {avg_rmse:.2f}%")
    print(f"Models with >55% Dir Acc: {(summary_df['dir_acc'] > 55).sum()} / {len(summary_df)}")
    
    print("\n✓ All enhanced models trained!")
    print(f"✓ Model files saved in: {enhanced_trainer.model_save_dir}/")
    print("✓ Performance metrics saved to database")
    print("\nIf directional accuracy > 55%, ready for Week 4: Trading System!")
    print("If still < 55%, we may need more feature engineering or longer sentiment history.")
else:
    print("No training summary available")