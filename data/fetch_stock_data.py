import requests
import pandas as pd
import time
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, api_key):
        """
        Initialize StockDataFetcher with Alpha Vantage API key
        
        Args:
            api_key (str): Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
    
    def _make_api_call(self, params: dict) -> Optional[dict]:
        """
        Make API call to Alpha Vantage with error handling
        
        Args:
            params (dict): API parameters
            
        Returns:
            Optional[dict]: API response data or None if failed
        """
        try:
            params['apikey'] = self.api_key
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error messages
            if "Error Message" in data:
                logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
                return None
            
            if "Note" in data:
                logger.warning(f"Alpha Vantage API Note: {data['Note']}")
                time.sleep(self.rate_limit_delay)
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a ticker over a specified period
        
        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
            period (str): Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical stock data or None if failed
        """
        logger.info(f"Fetching historical data for {ticker} with period {period}")
        
        # Map period to Alpha Vantage function
        if period in ["1d", "5d"]:
            function = "TIME_SERIES_INTRADAY"
            interval = "60min"
            params = {
                'function': function,
                'symbol': ticker,
                'interval': interval,
                'outputsize': 'full'
            }
        else:
            function = "TIME_SERIES_DAILY"
            params = {
                'function': function,
                'symbol': ticker,
                'outputsize': 'full'
            }
        
        data = self._make_api_call(params)
        if not data:
            return None
        
        try:
            if function == "TIME_SERIES_INTRADAY":
                time_series_key = f"Time Series ({interval})"
            else:
                time_series_key = "Time Series (Daily)"
            
            if time_series_key not in data:
                logger.error(f"Expected key '{time_series_key}' not found in response")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter by period if needed
            if period != "max":
                df = self._filter_by_period(df, period)
            
            logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing stock data for {ticker}: {e}")
            return None
    
    def get_real_time_stock_price(self, ticker: str) -> Optional[float]:
        """
        Get the current/latest stock price for a ticker
        
        Args:
            ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
            
        Returns:
            Optional[float]: Current stock price or None if failed
        """
        logger.info(f"Fetching real-time price for {ticker}")
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker
        }
        
        data = self._make_api_call(params)
        if not data:
            return None
        
        try:
            global_quote = data.get("Global Quote", {})
            if not global_quote:
                logger.error(f"No global quote data found for {ticker}")
                return None
            
            price = global_quote.get("05. price")
            if price:
                price_float = float(price)
                logger.info(f"Current price for {ticker}: ${price_float:.2f}")
                return price_float
            else:
                logger.error(f"Price not found in global quote for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing real-time price for {ticker}: {e}")
            return None
    
    def _filter_by_period(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """
        Filter DataFrame by specified period
        
        Args:
            df (pd.DataFrame): Stock data DataFrame
            period (str): Time period
            
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        now = pd.Timestamp.now()
        
        period_map = {
            "1d": now - pd.Timedelta(days=1),
            "5d": now - pd.Timedelta(days=5),
            "1mo": now - pd.Timedelta(days=30),
            "3mo": now - pd.Timedelta(days=90),
            "6mo": now - pd.Timedelta(days=180),
            "1y": now - pd.Timedelta(days=365),
            "2y": now - pd.Timedelta(days=730),
            "5y": now - pd.Timedelta(days=1825),
            "10y": now - pd.Timedelta(days=3650),
            "ytd": pd.Timestamp(now.year, 1, 1)
        }
        
        if period in period_map:
            start_date = period_map[period]
            return df[df.index >= start_date]
        
        return df
    
    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """
        Get company overview/information for a ticker
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[dict]: Company information or None if failed
        """
        logger.info(f"Fetching company info for {ticker}")
        
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker
        }
        
        data = self._make_api_call(params)
        if not data:
            return None
        
        try:
            # Filter out empty values
            company_info = {k: v for k, v in data.items() if v and v != "None"}
            logger.info(f"Successfully fetched company info for {ticker}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error processing company info for {ticker}: {e}")
            return None


# Convenience functions for backward compatibility
def get_stock_data(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data for a ticker over a specified period
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
        period (str): Time period
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with historical stock data or None if failed
    """
    fetcher = StockDataFetcher()
    return fetcher.get_stock_data(ticker, period)


def get_real_time_stock_price(ticker: str) -> Optional[float]:
    """
    Get the current stock price for a ticker
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Optional[float]: Current stock price or None if failed
    """
    fetcher = StockDataFetcher()
    return fetcher.get_real_time_stock_price(ticker)


# Example usage
if __name__ == "__main__":
    fetcher = StockDataFetcher()
    
    # Example: Get Apple stock data for 1 year
    print("Fetching AAPL data for 1 year...")
    aapl_data = fetcher.get_stock_data("AAPL", "1y")
    if aapl_data is not None:
        print(f"AAPL Data Shape: {aapl_data.shape}")
        print(aapl_data.head())
    
    # Example: Get real-time Apple stock price
    print("\nFetching real-time AAPL price...")
    aapl_price = fetcher.get_real_time_stock_price("AAPL")
    if aapl_price:
        print(f"AAPL Current Price: ${aapl_price:.2f}")
    
    # Example: Get company info
    print("\nFetching AAPL company info...")
    aapl_info = fetcher.get_stock_info("AAPL")
    if aapl_info:
        print(f"Company: {aapl_info.get('Name', 'N/A')}")
        print(f"Sector: {aapl_info.get('Sector', 'N/A')}")
        print(f"Market Cap: {aapl_info.get('MarketCapitalization', 'N/A')}") 