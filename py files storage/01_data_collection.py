# import dependencies
import yfinance as yf
import pandas as pd
import mysql.connector
from datetime import datetime, timedelta
import logging

#define YahooDataCollector class
class YahooDataCollector:
    def __init__(self, db_config):
        self.db_config = db_config
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_system.log'),
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
    
    def create_tables(self):
        """Create necessary database tables"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Stock information table
        stock_table = """
        CREATE TABLE IF NOT EXISTS stocks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10) UNIQUE NOT NULL,
            company_name VARCHAR(255),
            sector VARCHAR(100),
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Historical price data
        price_table = """
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            date DATE,
            open_price DECIMAL(10,4),
            high_price DECIMAL(10,4),
            low_price DECIMAL(10,4),
            close_price DECIMAL(10,4),
            volume BIGINT,
            adj_close DECIMAL(10,4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_symbol_date (symbol, date),
            FOREIGN KEY (symbol) REFERENCES stocks(symbol)
        )
        """
        
        # Simulated portfolio
        portfolio_table = """
        CREATE TABLE IF NOT EXISTS portfolio (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            shares_owned DECIMAL(10,4) DEFAULT 0,
            avg_cost_basis DECIMAL(10,4) DEFAULT 0,
            total_invested DECIMAL(10,2) DEFAULT 0,
            current_value DECIMAL(10,2) DEFAULT 0,
            unrealized_pnl DECIMAL(10,2) DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_symbol (symbol)
        )
        """
        
        # Simulated trades
        trades_table = """
        CREATE TABLE IF NOT EXISTS simulated_trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10),
            action ENUM('BUY', 'SELL'),
            shares DECIMAL(10,4),
            price DECIMAL(10,4),
            total_value DECIMAL(10,2),
            commission DECIMAL(10,2) DEFAULT 0,
            trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            strategy VARCHAR(50),
            notes TEXT
        )
        """
        
        # Account balance tracking
        account_table = """
        CREATE TABLE IF NOT EXISTS account_balance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cash_balance DECIMAL(12,2) DEFAULT 100000.00,
            total_portfolio_value DECIMAL(12,2) DEFAULT 0,
            total_account_value DECIMAL(12,2) DEFAULT 100000.00,
            daily_pnl DECIMAL(10,2) DEFAULT 0,
            total_pnl DECIMAL(10,2) DEFAULT 0,
            date DATE UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        tables = [stock_table, price_table, portfolio_table, trades_table, account_table]
        
        try:
            for table in tables:
                cursor.execute(table)
            conn.commit()
            self.logger.info("Database tables created successfully")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error creating tables: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def add_stock(self, symbol, company_name=None, sector=None):
        """Add a stock to tracking list"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        # Get company info from Yahoo Finance if not provided
        if not company_name:
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'Unknown')
            except:
                company_name = symbol
                sector = 'Unknown'
        
        try:
            query = """
            INSERT IGNORE INTO stocks (symbol, company_name, sector) 
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (symbol, company_name, sector))
            conn.commit()
            self.logger.info(f"Added stock: {symbol} - {company_name}")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error adding stock {symbol}: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def fetch_historical_data(self, symbol, period="2y"):
        """Fetch historical data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if hist.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
            
            # Reset index to get date as column
            hist.reset_index(inplace=True)
            hist['Symbol'] = symbol
            
            return hist
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def store_historical_data(self, symbol, data):
        """Store historical data in database"""
        if data is None or data.empty:
            return False
        
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        try:
            for _, row in data.iterrows():
                query = """
                INSERT IGNORE INTO historical_prices 
                (symbol, date, open_price, high_price, low_price, close_price, volume, adj_close)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    symbol,
                    row['Date'].date(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    float(row['Close'])  # Yahoo Finance 'Close' is already adjusted
                )
                cursor.execute(query, values)
            
            conn.commit()
            self.logger.info(f"Stored {len(data)} records for {symbol}")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def update_all_stocks(self):
        """Update historical data for all tracked stocks"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM stocks")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        success_count = 0
        for symbol in symbols:
            self.logger.info(f"Updating data for {symbol}")
            data = self.fetch_historical_data(symbol)
            if self.store_historical_data(symbol, data):
                success_count += 1
        
        self.logger.info(f"Updated {success_count}/{len(symbols)} stocks")
        return success_count == len(symbols)
    
    def get_current_price(self, symbol):
        """Get current/latest price for a symbol"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                # Fallback to daily data
                data = stock.history(period="5d")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def initialize_account(self, starting_balance=100000):
        """Initialize trading account with starting balance"""
        conn = self.connect_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        try:
            query = """
            INSERT IGNORE INTO account_balance 
            (cash_balance, total_portfolio_value, total_account_value, date)
            VALUES (%s, 0, %s, CURDATE())
            """
            cursor.execute(query, (starting_balance, starting_balance))
            conn.commit()
            self.logger.info(f"Initialized account with ${starting_balance:,.2f}")
            return True
        except mysql.connector.Error as e:
            self.logger.error(f"Error initializing account: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
            
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '',
        'database': 'trading_system'
    }
    
    # Initialize collector
    collector = YahooDataCollector(db_config)
    
    # Create database tables
    collector.create_tables()
    
    # Initialize account with $100,000 virtual money
    collector.initialize_account(100000)
    
    # 20 validated stocks
    stocks_to_track = [
        ('AAPL', 'Apple Inc.', 'Technology'),
        ('AMZN', 'Amazon.com, Inc.', 'Consumer Cyclical'),
        ('BLK', 'BlackRock, Inc.', 'Financial Services'),
        ('ERO', 'Ero Copper Corp.', 'Basic Materials'),
        ('FXP', 'ProShares UltraShort FTSE China 50', 'ETF'),
        ('GOOGL', 'Alphabet Inc.', 'Communication Services'),
        ('GXC', 'SPDR S&P China ETF', 'ETF'),
        ('JPM', 'JPMorgan Chase & Co.', 'Financial Services'),
        ('KR', 'The Kroger Co.', 'Consumer Defensive'),
        ('MDT', 'Medtronic plc', 'Healthcare'),
        ('META', 'Meta Platforms, Inc.', 'Communication Services'),
        ('MSFT', 'Microsoft Corporation', 'Technology'),
        ('NVDA', 'NVIDIA Corporation', 'Technology'),
        ('OXY', 'Occidental Petroleum Corporation', 'Energy'),
        ('PGJ', 'Invesco Golden Dragon China ETF', 'ETF'),
        ('RSP', 'Invesco S&P 500 Equal Weight ETF', 'ETF'),
        ('SPY', 'SPDR S&P 500 ETF', 'ETF'),    
    ]
    
    for symbol, name, sector in stocks_to_track:
        collector.add_stock(symbol, name, sector)
    
    # Fetch and store historical data
    print("Fetching historical data... This might take a few minutes...")
    collector.update_all_stocks()
    
    print("Setup complete! Check your database for the data.")