import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticFinancialDataGenerator:
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the synthetic financial data generator
        
        Args:
            seed (Optional[int]): Random seed for reproducible results
        """
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            Faker.seed(seed)
        
        self.fake = Faker()
        
        # Define realistic financial categories and patterns
        self.expense_categories = {
            'Housing': {'weight': 0.30, 'min_amount': 800, 'max_amount': 3000},
            'Food': {'weight': 0.15, 'min_amount': 200, 'max_amount': 800},
            'Transportation': {'weight': 0.12, 'min_amount': 100, 'max_amount': 600},
            'Utilities': {'weight': 0.08, 'min_amount': 80, 'max_amount': 300},
            'Healthcare': {'weight': 0.06, 'min_amount': 50, 'max_amount': 500},
            'Entertainment': {'weight': 0.08, 'min_amount': 50, 'max_amount': 400},
            'Shopping': {'weight': 0.10, 'min_amount': 100, 'max_amount': 600},
            'Insurance': {'weight': 0.05, 'min_amount': 100, 'max_amount': 400},
            'Savings': {'weight': 0.04, 'min_amount': 100, 'max_amount': 1000},
            'Miscellaneous': {'weight': 0.02, 'min_amount': 20, 'max_amount': 200}
        }
        
        self.income_sources = {
            'Salary': {'weight': 0.70, 'frequency': 'monthly'},
            'Freelance': {'weight': 0.15, 'frequency': 'irregular'},
            'Investment': {'weight': 0.08, 'frequency': 'quarterly'},
            'Side Business': {'weight': 0.05, 'frequency': 'monthly'},
            'Other': {'weight': 0.02, 'frequency': 'irregular'}
        }
        
        # Merchant names for realistic transactions
        self.merchants = {
            'Housing': ['Property Management Co', 'Landlord Services', 'Mortgage Corp'],
            'Food': ['Walmart', 'Target', 'Kroger', 'Whole Foods', 'Local Restaurant', 'Fast Food Chain'],
            'Transportation': ['Shell', 'BP', 'Exxon', 'Uber', 'Lyft', 'Metro Transit'],
            'Utilities': ['Electric Company', 'Gas Company', 'Water Department', 'Internet Provider'],
            'Healthcare': ['Medical Center', 'Pharmacy', 'Dental Office', 'Eye Care'],
            'Entertainment': ['Netflix', 'Spotify', 'Movie Theater', 'Concert Venue'],
            'Shopping': ['Amazon', 'Best Buy', 'Clothing Store', 'Department Store'],
            'Insurance': ['Auto Insurance', 'Health Insurance', 'Life Insurance'],
            'Savings': ['Savings Account', 'Investment Account', 'Retirement Fund'],
            'Miscellaneous': ['ATM Withdrawal', 'Bank Fee', 'Various']
        }
    
    def generate_user_profile(self) -> Dict:
        """
        Generate a realistic user financial profile
        
        Returns:
            Dict: User profile with demographics and financial characteristics
        """
        age = random.randint(22, 65)
        
        # Income based on age and career stage
        if age < 30:
            annual_income = random.randint(35000, 65000)
        elif age < 45:
            annual_income = random.randint(45000, 120000)
        else:
            annual_income = random.randint(55000, 150000)
        
        profile = {
            'user_id': self.fake.uuid4(),
            'name': self.fake.name(),
            'age': age,
            'annual_income': annual_income,
            'monthly_income': annual_income / 12,
            'location': f"{self.fake.city()}, {self.fake.state_abbr()}",
            'occupation': self.fake.job(),
            'credit_score': random.randint(580, 850),
            'savings_goal': random.randint(5000, 50000),
            'risk_tolerance': random.choice(['Conservative', 'Moderate', 'Aggressive']),
            'created_date': self.fake.date_between(start_date='-2y', end_date='today')
        }
        
        return profile
    
    def generate_transactions(self, 
                            user_profile: Dict, 
                            start_date: datetime = None, 
                            end_date: datetime = None, 
                            num_transactions: int = None) -> pd.DataFrame:
        """
        Generate realistic financial transactions for a user
        
        Args:
            user_profile (Dict): User profile from generate_user_profile()
            start_date (datetime): Start date for transactions
            end_date (datetime): End date for transactions
            num_transactions (int): Number of transactions to generate
            
        Returns:
            pd.DataFrame: DataFrame with transaction data
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        # Calculate realistic number of transactions if not provided
        if not num_transactions:
            days = (end_date - start_date).days
            num_transactions = int(days * random.uniform(2, 8))  # 2-8 transactions per day
        
        transactions = []
        monthly_income = user_profile['monthly_income']
        
        for _ in range(num_transactions):
            # Generate transaction date
            transaction_date = self.fake.date_between(start_date=start_date, end_date=end_date)
            
            # Determine if it's income or expense (90% expenses, 10% income)
            is_income = random.random() < 0.1
            
            if is_income:
                # Generate income transaction
                source = np.random.choice(
                    list(self.income_sources.keys()),
                    p=[source['weight'] for source in self.income_sources.values()]
                )
                
                if source == 'Salary':
                    amount = monthly_income + random.uniform(-monthly_income*0.1, monthly_income*0.1)
                elif source == 'Freelance':
                    amount = random.uniform(500, monthly_income * 0.5)
                elif source == 'Investment':
                    amount = random.uniform(100, monthly_income * 0.3)
                else:
                    amount = random.uniform(200, monthly_income * 0.2)
                
                transaction = {
                    'user_id': user_profile['user_id'],
                    'date': transaction_date,
                    'type': 'Income',
                    'category': source,
                    'amount': round(amount, 2),
                    'description': f"{source} - {self.fake.company()}",
                    'merchant': self.fake.company(),
                    'balance_after': 0  # Will be calculated later
                }
            else:
                # Generate expense transaction
                category = np.random.choice(
                    list(self.expense_categories.keys()),
                    p=[cat['weight'] for cat in self.expense_categories.values()]
                )
                
                cat_info = self.expense_categories[category]
                
                # Adjust expense amounts based on income level
                income_factor = user_profile['monthly_income'] / 5000  # Base factor
                min_amount = cat_info['min_amount'] * income_factor
                max_amount = cat_info['max_amount'] * income_factor
                
                amount = random.uniform(min_amount, max_amount)
                
                # Make some transactions more realistic (round numbers, etc.)
                if category in ['Housing', 'Insurance']:
                    amount = round(amount / 10) * 10  # Round to nearest 10
                elif random.random() < 0.3:
                    amount = round(amount)  # Round to nearest dollar
                
                merchant = random.choice(self.merchants[category])
                
                transaction = {
                    'user_id': user_profile['user_id'],
                    'date': transaction_date,
                    'type': 'Expense',
                    'category': category,
                    'amount': -round(amount, 2),  # Negative for expenses
                    'description': f"{category} - {merchant}",
                    'merchant': merchant,
                    'balance_after': 0  # Will be calculated later
                }
            
            transactions.append(transaction)
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate running balance
        df['balance_after'] = df['amount'].cumsum() + random.uniform(1000, 10000)  # Starting balance
        
        logger.info(f"Generated {len(df)} transactions for user {user_profile['user_id']}")
        return df
    
    def generate_monthly_summary(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate monthly financial summary from transactions
        
        Args:
            transactions_df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Monthly summary data
        """
        # Group by year-month
        monthly_data = transactions_df.copy()
        monthly_data['year_month'] = monthly_data['date'].dt.to_period('M')
        
        summary = monthly_data.groupby(['user_id', 'year_month']).agg({
            'amount': ['sum', 'count'],
            'date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['user_id', 'year_month', 'total_amount', 'transaction_count', 'first_date', 'last_date']
        
        # Calculate income and expenses separately
        income_summary = monthly_data[monthly_data['type'] == 'Income'].groupby(['user_id', 'year_month'])['amount'].sum().reset_index()
        income_summary.columns = ['user_id', 'year_month', 'total_income']
        
        expense_summary = monthly_data[monthly_data['type'] == 'Expense'].groupby(['user_id', 'year_month'])['amount'].sum().reset_index()
        expense_summary.columns = ['user_id', 'year_month', 'total_expenses']
        
        # Merge summaries
        summary = summary.merge(income_summary, on=['user_id', 'year_month'], how='left')
        summary = summary.merge(expense_summary, on=['user_id', 'year_month'], how='left')
        
        # Fill missing values
        summary['total_income'] = summary['total_income'].fillna(0)
        summary['total_expenses'] = summary['total_expenses'].fillna(0)
        
        # Calculate net savings
        summary['net_savings'] = summary['total_income'] + summary['total_expenses']  # expenses are negative
        
        logger.info(f"Generated monthly summary with {len(summary)} records")
        return summary
    
    def generate_investment_portfolio(self, user_profile: Dict, num_positions: int = None) -> pd.DataFrame:
        """
        Generate synthetic investment portfolio
        
        Args:
            user_profile (Dict): User profile
            num_positions (int): Number of investment positions
            
        Returns:
            pd.DataFrame: Investment portfolio data
        """
        if not num_positions:
            # Number of positions based on income and risk tolerance
            if user_profile['annual_income'] < 50000:
                num_positions = random.randint(3, 8)
            elif user_profile['annual_income'] < 100000:
                num_positions = random.randint(5, 15)
            else:
                num_positions = random.randint(8, 25)
        
        # Common stocks and ETFs
        securities = [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA',
            'SPY', 'QQQ', 'VTI', 'VEA', 'VWO', 'BND', 'VXUS', 'VOO',
            'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI'
        ]
        
        portfolio = []
        
        for _ in range(num_positions):
            security = random.choice(securities)
            
            # Determine asset type
            if security in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI']:
                asset_type = 'Cryptocurrency'
                shares = random.uniform(0.1, 10)
                current_price = random.uniform(100, 50000)
            elif security in ['SPY', 'QQQ', 'VTI', 'VEA', 'VWO', 'BND', 'VXUS', 'VOO']:
                asset_type = 'ETF'
                shares = random.randint(1, 100)
                current_price = random.uniform(50, 500)
            else:
                asset_type = 'Stock'
                shares = random.randint(1, 200)
                current_price = random.uniform(20, 3000)
            
            purchase_price = current_price * random.uniform(0.7, 1.3)  # Some variation
            purchase_date = self.fake.date_between(start_date='-2y', end_date='today')
            
            position = {
                'user_id': user_profile['user_id'],
                'symbol': security,
                'asset_type': asset_type,
                'shares': round(shares, 4),
                'purchase_price': round(purchase_price, 2),
                'current_price': round(current_price, 2),
                'purchase_date': purchase_date,
                'current_value': round(shares * current_price, 2),
                'total_return': round((current_price - purchase_price) * shares, 2),
                'return_percentage': round(((current_price - purchase_price) / purchase_price) * 100, 2)
            }
            
            portfolio.append(position)
        
        df = pd.DataFrame(portfolio)
        logger.info(f"Generated investment portfolio with {len(df)} positions")
        return df


def generate_sample_user_data(num_users: int = 5, 
                             transaction_days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate complete sample dataset for multiple users
    
    Args:
        num_users (int): Number of users to generate
        transaction_days (int): Number of days of transaction history
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
        (users, transactions, monthly_summaries, portfolios)
    """
    generator = SyntheticFinancialDataGenerator(seed=42)
    
    all_users = []
    all_transactions = []
    all_summaries = []
    all_portfolios = []
    
    start_date = datetime.now() - timedelta(days=transaction_days)
    end_date = datetime.now()
    
    for i in range(num_users):
        # Generate user profile
        user_profile = generator.generate_user_profile()
        all_users.append(user_profile)
        
        # Generate transactions
        transactions = generator.generate_transactions(
            user_profile, start_date, end_date
        )
        all_transactions.append(transactions)
        
        # Generate monthly summary
        monthly_summary = generator.generate_monthly_summary(transactions)
        all_summaries.append(monthly_summary)
        
        # Generate investment portfolio
        portfolio = generator.generate_investment_portfolio(user_profile)
        all_portfolios.append(portfolio)
        
        logger.info(f"Generated data for user {i+1}/{num_users}")
    
    # Combine all data
    users_df = pd.DataFrame(all_users)
    transactions_df = pd.concat(all_transactions, ignore_index=True)
    summaries_df = pd.concat(all_summaries, ignore_index=True)
    portfolios_df = pd.concat(all_portfolios, ignore_index=True)
    
    return users_df, transactions_df, summaries_df, portfolios_df


# Example usage
if __name__ == "__main__":
    # Generate sample data
    print("Generating synthetic financial data...")
    
    users, transactions, summaries, portfolios = generate_sample_user_data(
        num_users=3, transaction_days=180
    )
    
    print(f"\nGenerated data summary:")
    print(f"Users: {len(users)}")
    print(f"Transactions: {len(transactions)}")
    print(f"Monthly summaries: {len(summaries)}")
    print(f"Portfolio positions: {len(portfolios)}")
    
    print(f"\nSample user profile:")
    print(users.iloc[0].to_dict())
    
    print(f"\nSample transactions:")
    print(transactions.head())
    
    print(f"\nSample portfolio:")
    print(portfolios.head()) 