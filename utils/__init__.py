"""
Pecunia AI - Enterprise Utilities Package
Advanced financial data processing, authentication, API management, and blockchain integration
"""

from .api_utils import (
    APIManager, EnterpriseAPIClient, WebhookHandler, CircuitBreaker,
    DistributedRateLimiter, MultiTierCache, api_manager, api_endpoint
)

from .auth import (
    EnterpriseAuthManager, MFAManager, JWTManager, BiometricAuthenticator,
    OAuth2Provider, SecurityAnalyzer, AuthMethod, UserRole, SecurityPolicy,
    require_auth, require_mfa, initialize_auth_system
)

from .data_cleaning import (
    AdvancedDataCleaner, StreamProcessor, RealTimeAnomalyMonitor,
    StatisticalAnomalyDetector, MLAnomalyDetector, TimeSeriesAnomalyDetector,
    DataQualityAssessor, DataQuality, AnomalyType, ProcessingMode,
    clean_data, detect_anomalies, initialize_data_processing
)

from .blockchain import (
    BlockchainIntegrationManager, NetworkManager, PriceOracle, 
    DeFiProtocolManager, NFTPortfolioTracker, BlockchainNetwork,
    DeFiProtocol, NFTStandard, initialize_blockchain_integration,
    get_wallet_portfolio, get_token_price, get_nft_collection
)

__version__ = "1.0.0"
__author__ = "Pecunia AI Team"

# Global managers (to be initialized)
api_manager = None
auth_manager = None
data_cleaner = None
blockchain_manager = None

def initialize_all_systems(
    database_url: str = "sqlite:///pecunia.db",
    redis_url: str = "redis://localhost:6379",
    kafka_config: dict = None,
    security_policy: SecurityPolicy = None
):
    """
    Initialize all utility systems with common configuration
    
    Args:
        database_url: Database connection URL
        redis_url: Redis connection URL  
        kafka_config: Kafka configuration for streaming
        security_policy: Security policy configuration
        
    Returns:
        dict: Initialized system managers
    """
    global api_manager, auth_manager, data_cleaner, blockchain_manager
    
    # Initialize API management
    from .api_utils import APIManager
    api_manager = APIManager(redis_url)
    
    # Initialize authentication system
    auth_manager = initialize_auth_system(database_url, redis_url, security_policy)
    
    # Initialize data processing
    stream_processor, anomaly_monitor = initialize_data_processing(redis_url, kafka_config)
    
    # Initialize blockchain integration
    blockchain_manager = initialize_blockchain_integration(redis_url)
    
    return {
        'api_manager': api_manager,
        'auth_manager': auth_manager,
        'stream_processor': stream_processor,
        'anomaly_monitor': anomaly_monitor,
        'blockchain_manager': blockchain_manager
    }

__all__ = [
    # API Utils
    'APIManager', 'EnterpriseAPIClient', 'WebhookHandler', 'CircuitBreaker',
    'DistributedRateLimiter', 'MultiTierCache', 'api_manager', 'api_endpoint',
    
    # Authentication
    'EnterpriseAuthManager', 'MFAManager', 'JWTManager', 'BiometricAuthenticator',
    'OAuth2Provider', 'SecurityAnalyzer', 'AuthMethod', 'UserRole', 'SecurityPolicy',
    'require_auth', 'require_mfa', 'initialize_auth_system',
    
    # Data Processing
    'AdvancedDataCleaner', 'StreamProcessor', 'RealTimeAnomalyMonitor',
    'StatisticalAnomalyDetector', 'MLAnomalyDetector', 'TimeSeriesAnomalyDetector',
    'DataQualityAssessor', 'DataQuality', 'AnomalyType', 'ProcessingMode',
    'clean_data', 'detect_anomalies', 'initialize_data_processing',
    
    # Blockchain
    'BlockchainIntegrationManager', 'NetworkManager', 'PriceOracle', 
    'DeFiProtocolManager', 'NFTPortfolioTracker', 'BlockchainNetwork',
    'DeFiProtocol', 'NFTStandard', 'initialize_blockchain_integration',
    'get_wallet_portfolio', 'get_token_price', 'get_nft_collection',
    
    # Global functions
    'initialize_all_systems'
] 