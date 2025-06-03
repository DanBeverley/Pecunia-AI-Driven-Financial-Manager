import pandas as pd
import sqlite3
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataStorage:
    def __init__(self, db_path: str = "pecunia_database.db", csv_dir: str = "data_exports"):
        """
        Initialize DataStorage for managing CSV and SQLite data storage
        
        Args:
            db_path (str): Path to SQLite database file
            csv_dir (str): Directory for CSV exports
        """
        self.db_path = db_path
        self.csv_dir = Path(csv_dir)
        
        self.csv_dir.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        age INTEGER,
                        annual_income REAL,
                        monthly_income REAL,
                        location TEXT,
                        occupation TEXT,
                        credit_score INTEGER,
                        savings_goal REAL,
                        risk_tolerance TEXT,
                        created_date DATE,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Transactions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        amount REAL NOT NULL,
                        description TEXT,
                        merchant TEXT,
                        balance_after REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Monthly summaries table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS monthly_summaries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        year_month TEXT NOT NULL,
                        total_amount REAL,
                        transaction_count INTEGER,
                        first_date DATE,
                        last_date DATE,
                        total_income REAL,
                        total_expenses REAL,
                        net_savings REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        UNIQUE(user_id, year_month)
                    )
                ''')
                
                # Investment portfolios table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS portfolios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        shares REAL NOT NULL,
                        purchase_price REAL NOT NULL,
                        current_price REAL NOT NULL,
                        purchase_date DATE NOT NULL,
                        current_value REAL,
                        total_return REAL,
                        return_percentage REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
                # Stock data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stock_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume INTEGER,
                        data_source TEXT DEFAULT 'Alpha Vantage',
                        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                ''')
                
                # Crypto data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS crypto_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        close_price REAL,
                        volume REAL,
                        market_cap REAL,
                        data_source TEXT DEFAULT 'CoinMarketCap',
                        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                ''')
                
                # Real-time prices table (for current prices)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS current_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        current_price REAL NOT NULL,
                        data_source TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, asset_type)
                    )
                ''')
                
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_data_symbol_date ON stock_data(symbol, date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_crypto_data_symbol_date ON crypto_data(symbol, date)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id)')
                
                conn.commit()
                logger.info(f"Database initialized successfully: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_to_csv(self, data: pd.DataFrame, filename: str) -> bool:
        """
        Save DataFrame to CSV file
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Name of the CSV file (without .csv extension)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Add timestamp to filename if not present
            if not filename.endswith('.csv'):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{filename}_{timestamp}.csv"
            
            filepath = self.csv_dir / filename
            data.to_csv(filepath, index=False)
            
            logger.info(f"Data saved to CSV: {filepath} ({len(data)} records)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to CSV {filename}: {e}")
            return False
    
    def save_to_db(self, data: pd.DataFrame, table_name: str, if_exists: str = 'append') -> bool:
        """
        Save DataFrame to SQLite database
        
        Args:
            data (pd.DataFrame): Data to save
            table_name (str): Name of the database table
            if_exists (str): What to do if table exists ('append', 'replace', 'fail')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Handle special formatting for certain tables
                if table_name in ['monthly_summaries']:
                    # Convert period to string for monthly summaries
                    data_copy = data.copy()
                    if 'year_month' in data_copy.columns:
                        data_copy['year_month'] = data_copy['year_month'].astype(str)
                    data = data_copy
                
                data.to_sql(table_name, conn, if_exists=if_exists, index=False)
                
                logger.info(f"Data saved to database table '{table_name}': {len(data)} records")
                return True
                
        except Exception as e:
            logger.error(f"Error saving to database table {table_name}: {e}")
            return False
    
    def load_from_db(self, table_name: str, where_clause: str = None, params: tuple = None) -> Optional[pd.DataFrame]:
        """
        Load data from SQLite database
        
        Args:
            table_name (str): Name of the database table
            where_clause (str): Optional WHERE clause (without WHERE keyword)
            params (tuple): Parameters for the WHERE clause
            
        Returns:
            Optional[pd.DataFrame]: Loaded data or None if failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = f"SELECT * FROM {table_name}"
                
                if where_clause:
                    query += f" WHERE {where_clause}"
                
                data = pd.read_sql_query(query, conn, params=params)
                
                logger.info(f"Loaded {len(data)} records from table '{table_name}'")
                return data
                
        except Exception as e:
            logger.error(f"Error loading from database table {table_name}: {e}")
            return None
    
    def save_stock_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Save stock data with proper formatting
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            symbol (str): Stock symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for database
            data_copy = data.copy()
            data_copy.reset_index(inplace=True)
            data_copy['symbol'] = symbol
            
            # Rename columns to match database schema
            column_mapping = {
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume',
                'index': 'date'  # if index was datetime
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data_copy.columns:
                    data_copy.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure date column is properly formatted
            if 'date' in data_copy.columns:
                data_copy['date'] = pd.to_datetime(data_copy['date']).dt.date
            
            return self.save_to_db(data_copy, 'stock_data', if_exists='replace')
            
        except Exception as e:
            logger.error(f"Error saving stock data for {symbol}: {e}")
            return False
    
    def save_crypto_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Save crypto data with proper formatting
        
        Args:
            data (pd.DataFrame): Crypto data with OHLCV columns
            symbol (str): Crypto symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare data for database
            data_copy = data.copy()
            data_copy.reset_index(inplace=True)
            data_copy['symbol'] = symbol
            
            # Rename columns to match database schema
            column_mapping = {
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume',
                'Market_Cap': 'market_cap',
                'index': 'date'  # if index was datetime
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in data_copy.columns:
                    data_copy.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure date column is properly formatted
            if 'date' in data_copy.columns:
                data_copy['date'] = pd.to_datetime(data_copy['date']).dt.date
            
            return self.save_to_db(data_copy, 'crypto_data', if_exists='replace')
            
        except Exception as e:
            logger.error(f"Error saving crypto data for {symbol}: {e}")
            return False
    
    def save_current_price(self, symbol: str, asset_type: str, price: float, data_source: str) -> bool:
        """
        Save or update current price for an asset
        
        Args:
            symbol (str): Asset symbol
            asset_type (str): 'stock' or 'crypto'
            price (float): Current price
            data_source (str): Data source name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO current_prices 
                    (symbol, asset_type, current_price, data_source, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (symbol, asset_type, price, data_source))
                
                conn.commit()
                logger.info(f"Updated current price for {symbol} ({asset_type}): ${price:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving current price for {symbol}: {e}")
            return False
    
    def get_latest_data(self, symbol: str, asset_type: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get latest data for a symbol from database
        
        Args:
            symbol (str): Asset symbol
            asset_type (str): 'stock' or 'crypto'
            days (int): Number of days of data to retrieve
            
        Returns:
            Optional[pd.DataFrame]: Latest data or None if not found
        """
        try:
            table_name = 'stock_data' if asset_type == 'stock' else 'crypto_data'
            
            where_clause = f"symbol = ? AND date >= date('now', '-{days} days')"
            params = (symbol,)
            
            data = self.load_from_db(table_name, where_clause, params)
            
            if data is not None and len(data) > 0:
                # Convert date column back to datetime and set as index
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                data.sort_index(inplace=True)
                
                logger.info(f"Retrieved {len(data)} records for {symbol} from database")
                return data
            else:
                logger.info(f"No data found for {symbol} in database")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def backup_database(self, backup_path: str = None) -> bool:
        """
        Create a backup of the database
        
        Args:
            backup_path (str): Path for backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"pecunia_backup_{timestamp}.db"
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        stats = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table counts
                tables = ['users', 'transactions', 'monthly_summaries', 'portfolios', 
                         'stock_data', 'crypto_data', 'current_prices']
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[f"{table}_count"] = count
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                size_bytes = cursor.fetchone()[0]
                stats['database_size_mb'] = round(size_bytes / (1024 * 1024), 2)
                
                # Get unique symbols
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stock_data")
                stats['unique_stocks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT symbol) FROM crypto_data")
                stats['unique_cryptos'] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
        
        return stats


# Convenience functions for backward compatibility
def save_to_csv(data: pd.DataFrame, filename: str) -> bool:
    """
    Save DataFrame to CSV file
    
    Args:
        data (pd.DataFrame): Data to save
        filename (str): Name of the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    storage = DataStorage()
    return storage.save_to_csv(data, filename)


def save_to_db(data: pd.DataFrame, table_name: str) -> bool:
    """
    Save DataFrame to SQLite database
    
    Args:
        data (pd.DataFrame): Data to save
        table_name (str): Name of the database table
        
    Returns:
        bool: True if successful, False otherwise
    """
    storage = DataStorage()
    return storage.save_to_db(data, table_name)


# Example usage
if __name__ == "__main__":
    # Initialize storage
    storage = DataStorage()
    
    # Example: Generate and save some synthetic data
    from generate_synthetic_data import generate_sample_user_data
    
    print("Generating sample data...")
    users, transactions, summaries, portfolios = generate_sample_user_data(num_users=2, transaction_days=90)
    
    print("Saving data to database...")
    
    # Save to database
    storage.save_to_db(users, 'users')
    storage.save_to_db(transactions, 'transactions')
    storage.save_to_db(summaries, 'monthly_summaries')
    storage.save_to_db(portfolios, 'portfolios')
    
    # Save to CSV
    storage.save_to_csv(users, 'sample_users')
    storage.save_to_csv(transactions, 'sample_transactions')
    
    # Get database statistics
    stats = storage.get_database_stats()
    print(f"\nDatabase Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example: Load data back
    loaded_users = storage.load_from_db('users')
    print(f"\nLoaded {len(loaded_users)} users from database")
    
    # Create backup
    storage.backup_database()
    print("Database backup created") 