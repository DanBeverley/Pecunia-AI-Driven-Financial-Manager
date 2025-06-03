import requests
import pandas as pd
import time
from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    def __init__(self, api_key: str = "b7e04efd-0150-4825-8a3b-c6daf473b1d1"):
        """
        Initialize CryptoDataFetcher with CoinMarketCap API key
        
        Args:
            api_key (str): CoinMarketCap API key
        """
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': self.api_key
        }
        self.rate_limit_delay = 1  # CoinMarketCap allows more frequent calls
        
        # Cache for coin ID mapping
        self._coin_map_cache = {}
    
    def _make_api_call(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """
        Make API call to CoinMarketCap with error handling
        
        Args:
            endpoint (str): API endpoint
            params (dict): API parameters
            
        Returns:
            Optional[dict]: API response data or None if failed
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error
            if data.get('status', {}).get('error_code') != 0:
                error_msg = data.get('status', {}).get('error_message', 'Unknown error')
                logger.error(f"CoinMarketCap API Error: {error_msg}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    def _get_coin_id(self, coin_symbol_or_name: str) -> Optional[int]:
        """
        Get CoinMarketCap coin ID from symbol or name
        
        Args:
            coin_symbol_or_name (str): Coin symbol (e.g., "BTC") or name (e.g., "Bitcoin")
            
        Returns:
            Optional[int]: Coin ID or None if not found
        """
        if coin_symbol_or_name in self._coin_map_cache:
            return self._coin_map_cache[coin_symbol_or_name]
        
        # Try to get coin map
        data = self._make_api_call("cryptocurrency/map")
        if not data:
            return None
        
        coin_symbol_or_name_lower = coin_symbol_or_name.lower()
        
        for coin in data.get('data', []):
            symbol = coin.get('symbol', '').lower()
            name = coin.get('name', '').lower()
            coin_id = coin.get('id')
            
            if symbol == coin_symbol_or_name_lower or name == coin_symbol_or_name_lower:
                self._coin_map_cache[coin_symbol_or_name] = coin_id
                return coin_id
        
        logger.error(f"Coin '{coin_symbol_or_name}' not found")
        return None
    
    def get_crypto_data(self, coin_id: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch historical cryptocurrency data for a coin over specified days
        
        Args:
            coin_id (str): Coin symbol or name (e.g., "BTC", "Bitcoin", "ETH", "Ethereum")
            days (int): Number of days of historical data to fetch
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with historical crypto data or None if failed
        """
        logger.info(f"Fetching historical data for {coin_id} for {days} days")
        
        # Get coin ID
        coin_cmc_id = self._get_coin_id(coin_id)
        if not coin_cmc_id:
            return None
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        params = {
            'id': coin_cmc_id,
            'time_start': start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'time_end': end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'interval': self._get_interval_from_days(days)
        }
        
        data = self._make_api_call("cryptocurrency/quotes/historical", params)
        if not data:
            return None
        
        try:
            quotes = data.get('data', {}).get('quotes', [])
            if not quotes:
                logger.error(f"No historical data found for {coin_id}")
                return None
            
            # Process data into DataFrame
            rows = []
            for quote in quotes:
                timestamp = quote.get('timestamp')
                quote_data = quote.get('quote', {}).get('USD', {})
                
                if timestamp and quote_data:
                    rows.append({
                        'timestamp': pd.to_datetime(timestamp),
                        'open': quote_data.get('open'),
                        'high': quote_data.get('high'),
                        'low': quote_data.get('low'),
                        'close': quote_data.get('close'),
                        'volume': quote_data.get('volume'),
                        'market_cap': quote_data.get('market_cap')
                    })
            
            if not rows:
                logger.error(f"No valid data points found for {coin_id}")
                return None
            
            df = pd.DataFrame(rows)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Rename columns to match stock data format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Market_Cap']
            
            logger.info(f"Successfully fetched {len(df)} records for {coin_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing crypto data for {coin_id}: {e}")
            return None
    
    def get_real_time_crypto_price(self, coin_id: str) -> Optional[float]:
        """
        Get the current price of a cryptocurrency
        
        Args:
            coin_id (str): Coin symbol or name (e.g., "BTC", "Bitcoin", "ETH", "Ethereum")
            
        Returns:
            Optional[float]: Current crypto price in USD or None if failed
        """
        logger.info(f"Fetching real-time price for {coin_id}")
        
        # Get coin ID
        coin_cmc_id = self._get_coin_id(coin_id)
        if not coin_cmc_id:
            return None
        
        params = {
            'id': coin_cmc_id,
            'convert': 'USD'
        }
        
        data = self._make_api_call("cryptocurrency/quotes/latest", params)
        if not data:
            return None
        
        try:
            coin_data = data.get('data', {}).get(str(coin_cmc_id), {})
            if not coin_data:
                logger.error(f"No price data found for {coin_id}")
                return None
            
            quote = coin_data.get('quote', {}).get('USD', {})
            price = quote.get('price')
            
            if price is not None:
                logger.info(f"Current price for {coin_id}: ${price:.2f}")
                return float(price)
            else:
                logger.error(f"Price not found for {coin_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing real-time price for {coin_id}: {e}")
            return None
    
    def get_crypto_info(self, coin_id: str) -> Optional[dict]:
        """
        Get detailed information about a cryptocurrency
        
        Args:
            coin_id (str): Coin symbol or name
            
        Returns:
            Optional[dict]: Crypto information or None if failed
        """
        logger.info(f"Fetching crypto info for {coin_id}")
        
        # Get coin ID
        coin_cmc_id = self._get_coin_id(coin_id)
        if not coin_cmc_id:
            return None
        
        params = {
            'id': coin_cmc_id
        }
        
        data = self._make_api_call("cryptocurrency/info", params)
        if not data:
            return None
        
        try:
            coin_info = data.get('data', {}).get(str(coin_cmc_id), {})
            if coin_info:
                logger.info(f"Successfully fetched info for {coin_id}")
                return coin_info
            else:
                logger.error(f"No info found for {coin_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing crypto info for {coin_id}: {e}")
            return None
    
    def get_top_cryptos(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Get top cryptocurrencies by market cap
        
        Args:
            limit (int): Number of top cryptos to fetch (max 5000)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with top crypto data or None if failed
        """
        logger.info(f"Fetching top {limit} cryptocurrencies")
        
        params = {
            'start': 1,
            'limit': limit,
            'convert': 'USD'
        }
        
        data = self._make_api_call("cryptocurrency/listings/latest", params)
        if not data:
            return None
        
        try:
            cryptos = data.get('data', [])
            if not cryptos:
                logger.error("No crypto data found")
                return None
            
            rows = []
            for crypto in cryptos:
                quote = crypto.get('quote', {}).get('USD', {})
                rows.append({
                    'symbol': crypto.get('symbol'),
                    'name': crypto.get('name'),
                    'price': quote.get('price'),
                    'volume_24h': quote.get('volume_24h'),
                    'market_cap': quote.get('market_cap'),
                    'percent_change_1h': quote.get('percent_change_1h'),
                    'percent_change_24h': quote.get('percent_change_24h'),
                    'percent_change_7d': quote.get('percent_change_7d'),
                    'cmc_rank': crypto.get('cmc_rank')
                })
            
            df = pd.DataFrame(rows)
            logger.info(f"Successfully fetched top {len(df)} cryptocurrencies")
            return df
            
        except Exception as e:
            logger.error(f"Error processing top cryptos: {e}")
            return None
    
    def _get_interval_from_days(self, days: int) -> str:
        """
        Get appropriate interval based on number of days
        
        Args:
            days (int): Number of days
            
        Returns:
            str: Interval string for API
        """
        if days <= 1:
            return "5m"
        elif days <= 7:
            return "1h"
        elif days <= 30:
            return "2h"
        elif days <= 90:
            return "6h"
        elif days <= 365:
            return "1d"
        else:
            return "1d"


# Convenience functions for backward compatibility
def get_crypto_data(coin_id: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch historical cryptocurrency data for a coin over specified days
    
    Args:
        coin_id (str): Coin symbol or name (e.g., "BTC", "Bitcoin")
        days (int): Number of days of historical data
        
    Returns:
        Optional[pd.DataFrame]: DataFrame with historical crypto data or None if failed
    """
    fetcher = CryptoDataFetcher()
    return fetcher.get_crypto_data(coin_id, days)


def get_real_time_crypto_price(coin_id: str) -> Optional[float]:
    """
    Get the current price of a cryptocurrency
    
    Args:
        coin_id (str): Coin symbol or name
        
    Returns:
        Optional[float]: Current crypto price in USD or None if failed
    """
    fetcher = CryptoDataFetcher()
    return fetcher.get_real_time_crypto_price(coin_id)


# Example usage
if __name__ == "__main__":
    # Initialize fetcher
    fetcher = CryptoDataFetcher()
    
    # Example: Get Bitcoin data for 30 days
    print("Fetching BTC data for 30 days...")
    btc_data = fetcher.get_crypto_data("BTC", 30)
    if btc_data is not None:
        print(f"BTC Data Shape: {btc_data.shape}")
        print(btc_data.head())
    
    # Example: Get real-time Bitcoin price
    print("\nFetching real-time BTC price...")
    btc_price = fetcher.get_real_time_crypto_price("BTC")
    if btc_price:
        print(f"BTC Current Price: ${btc_price:.2f}")
    
    # Example: Get top 5 cryptocurrencies
    print("\nFetching top 5 cryptocurrencies...")
    top_cryptos = fetcher.get_top_cryptos(5)
    if top_cryptos is not None:
        print(top_cryptos[['symbol', 'name', 'price', 'market_cap']].head())
    
    # Example: Get Ethereum info
    print("\nFetching ETH info...")
    eth_info = fetcher.get_crypto_info("ETH")
    if eth_info:
        print(f"Name: {eth_info.get('name', 'N/A')}")
        print(f"Description: {eth_info.get('description', 'N/A')[:100]}...")
        print(f"Website: {eth_info.get('urls', {}).get('website', ['N/A'])[0]}") 