"""
Pecunia AI - Enterprise API Management System
Advanced API utilities with distributed rate limiting, circuit breakers, and webhook handling
"""

import asyncio
import hashlib
import hmac
import json
import time
import redis
import requests
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps, lru_cache
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from urllib.parse import urljoin, urlparse
import concurrent.futures
import queue
import threading

from flask import Flask, request, jsonify, abort
from cryptography.fernet import Fernet
import jwt
from celery import Celery
import backoff

# Configure sophisticated logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CacheStrategy(Enum):
    """Advanced caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class APIEndpoint:
    """Enterprise API endpoint configuration"""
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_attempts: int = 3
    cache_ttl: int = 300
    rate_limit: int = 100
    circuit_breaker_threshold: int = 5
    auth_required: bool = True

@dataclass
class RateLimitConfig:
    """Distributed rate limiting configuration"""
    requests_per_window: int
    window_size: int
    burst_limit: int
    key_prefix: str = "rate_limit"
    storage_backend: str = "redis"

class CircuitBreaker:
    """Advanced circuit breaker with exponential backoff"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = RLock()
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == CircuitBreakerState.OPEN:
                    if self._should_attempt_reset():
                        self.state = CircuitBreakerState.HALF_OPEN
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except self.expected_exception as e:
                    self._on_failure()
                    raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class DistributedRateLimiter:
    """Enterprise distributed rate limiter with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis, config: RateLimitConfig):
        self.redis = redis_client
        self.config = config
        self.lua_script = self._load_lua_script()
    
    def _load_lua_script(self) -> str:
        """Advanced Lua script for atomic rate limiting operations"""
        return """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local burst = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local current = redis.call('GET', key)
        if current == false then
            current = 0
        else
            current = tonumber(current)
        end
        
        local window_start = now - window
        local expired_key = key .. ':' .. window_start
        redis.call('DEL', expired_key)
        
        if current < limit then
            redis.call('INCR', key)
            redis.call('EXPIRE', key, window)
            return {1, limit - current - 1}
        elseif current < burst then
            redis.call('INCR', key)
            redis.call('EXPIRE', key, window)
            return {1, burst - current - 1}
        else
            return {0, 0}
        """
    
    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """Check if request is allowed under rate limit"""
        key = f"{self.config.key_prefix}:{identifier}"
        now = int(time.time())
        
        try:
            result = self.redis.eval(
                self.lua_script, 1, key,
                self.config.window_size,
                self.config.requests_per_window,
                self.config.burst_limit,
                now
            )
            return bool(result[0]), int(result[1])
        except redis.RedisError as e:
            logger.error(f"Rate limiter Redis error: {e}")
            return True, 0  # Fail open

class MultiTierCache:
    """Advanced multi-tier caching system"""
    
    def __init__(self, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.strategy = strategy
        self.local_cache = {}
        self.access_times = defaultdict(list)
        self.access_counts = defaultdict(int)
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.local_cache:
                self._record_access(key)
                self.cache_stats["hits"] += 1
                return self.local_cache[key]["data"]
            
            self.cache_stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        with self._lock:
            if len(self.local_cache) >= 1000:  # Max cache size
                self._evict_key()
            
            self.local_cache[key] = {
                "data": value,
                "timestamp": time.time(),
                "ttl": ttl,
                "access_count": 1
            }
            self._record_access(key)
    
    def _record_access(self, key: str):
        now = time.time()
        self.access_times[key].append(now)
        self.access_counts[key] += 1
        
        # Keep only recent access times
        self.access_times[key] = [t for t in self.access_times[key] if now - t < 3600]
    
    def _evict_key(self):
        """Intelligent key eviction based on strategy"""
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = min(self.local_cache.keys(), 
                           key=lambda k: self.access_times[k][-1] if self.access_times[k] else 0)
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            oldest_key = min(self.local_cache.keys(), 
                           key=lambda k: self.access_counts[k])
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            oldest_key = self._adaptive_eviction()
        else:
            # FIFO fallback
            oldest_key = next(iter(self.local_cache))
        
        del self.local_cache[oldest_key]
        self.cache_stats["evictions"] += 1
    
    def _adaptive_eviction(self) -> str:
        """Advanced adaptive eviction algorithm"""
        scores = {}
        now = time.time()
        
        for key in self.local_cache.keys():
            age = now - self.local_cache[key]["timestamp"]
            frequency = self.access_counts[key]
            recency = now - (self.access_times[key][-1] if self.access_times[key] else 0)
            
            # Composite score: lower is better for eviction
            scores[key] = (age * 0.3) + (1/frequency * 0.4) + (recency * 0.3)
        
        return max(scores.keys(), key=lambda k: scores[k])

class WebhookHandler:
    """Enterprise webhook processing system"""
    
    def __init__(self, secret_key: str, redis_client: redis.Redis):
        self.secret_key = secret_key
        self.redis = redis_client
        self.app = Flask(__name__)
        self.rate_limiter = DistributedRateLimiter(
            redis_client, 
            RateLimitConfig(requests_per_window=100, window_size=60, burst_limit=150)
        )
        self.message_queue = queue.Queue(maxsize=10000)
        self.worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self._setup_routes()
        self._start_workers()
    
    def _setup_routes(self):
        """Setup Flask routes for webhook endpoints"""
        
        @self.app.route('/webhook/financial-data', methods=['POST'])
        def handle_financial_webhook():
            return self._process_webhook('financial_data')
        
        @self.app.route('/webhook/market-data', methods=['POST'])
        def handle_market_webhook():
            return self._process_webhook('market_data')
        
        @self.app.route('/webhook/alerts', methods=['POST'])
        def handle_alerts_webhook():
            return self._process_webhook('alerts')
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})
    
    def _process_webhook(self, webhook_type: str):
        """Process incoming webhook with security and rate limiting"""
        client_ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        
        # Rate limiting
        allowed, remaining = self.rate_limiter.is_allowed(client_ip)
        if not allowed:
            abort(429, description="Rate limit exceeded")
        
        # Signature verification
        if not self._verify_signature(request):
            abort(401, description="Invalid signature")
        
        try:
            payload = request.get_json(force=True)
            if not payload:
                abort(400, description="Invalid JSON payload")
            
            # Async processing
            webhook_data = {
                "type": webhook_type,
                "payload": payload,
                "timestamp": time.time(),
                "source_ip": client_ip,
                "headers": dict(request.headers)
            }
            
            self.message_queue.put(webhook_data, timeout=1)
            
            return jsonify({
                "status": "accepted",
                "message_id": hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
            }), 202
            
        except queue.Full:
            abort(503, description="Service temporarily unavailable")
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            abort(500, description="Internal server error")
    
    def _verify_signature(self, request) -> bool:
        """Verify webhook signature for security"""
        signature = request.headers.get('X-Webhook-Signature')
        if not signature:
            return False
        
        expected_signature = hmac.new(
            self.secret_key.encode(),
            request.get_data(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def _start_workers(self):
        """Start background workers for async processing"""
        for _ in range(5):
            self.worker_pool.submit(self._worker_loop)
    
    def _worker_loop(self):
        """Worker loop for processing webhook messages"""
        while True:
            try:
                webhook_data = self.message_queue.get(timeout=5)
                self._process_webhook_data(webhook_data)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_webhook_data(self, webhook_data: Dict[str, Any]):
        """Process webhook data based on type"""
        webhook_type = webhook_data["type"]
        payload = webhook_data["payload"]
        
        if webhook_type == "financial_data":
            self._handle_financial_data(payload)
        elif webhook_type == "market_data":
            self._handle_market_data(payload)
        elif webhook_type == "alerts":
            self._handle_alerts(payload)
    
    def _handle_financial_data(self, payload: Dict[str, Any]):
        """Handle financial data webhooks"""
        # Store in Redis for real-time access
        key = f"financial_data:{payload.get('symbol', 'unknown')}"
        self.redis.setex(key, 300, json.dumps(payload))
        logger.info(f"Processed financial data: {payload.get('symbol')}")
    
    def _handle_market_data(self, payload: Dict[str, Any]):
        """Handle market data webhooks"""
        # Stream processing for market data
        key = f"market_data:{int(time.time())}"
        self.redis.lpush("market_stream", json.dumps(payload))
        self.redis.expire("market_stream", 3600)
        logger.info(f"Processed market data: {len(payload)} data points")
    
    def _handle_alerts(self, payload: Dict[str, Any]):
        """Handle alert webhooks"""
        # Priority queue for alerts
        priority = payload.get('priority', 'medium')
        alert_key = f"alerts:{priority}"
        self.redis.lpush(alert_key, json.dumps(payload))
        logger.info(f"Processed alert: {payload.get('type')}")

class EnterpriseAPIClient:
    """Sophisticated API client with advanced features"""
    
    def __init__(self, base_url: str, api_key: str, redis_client: redis.Redis):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.cache = MultiTierCache(CacheStrategy.ADAPTIVE)
        self.rate_limiter = DistributedRateLimiter(
            redis_client,
            RateLimitConfig(requests_per_window=1000, window_size=3600, burst_limit=1500)
        )
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._setup_session()
    
    def _setup_session(self):
        """Configure session with advanced settings"""
        self.session.headers.update({
            'User-Agent': 'Pecunia-AI/1.0',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        })
        
        # Connection pooling and retry strategy
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    @backoff.on_exception(backoff.expo, requests.RequestException, max_tries=3)
    def make_request(self, endpoint: APIEndpoint, **kwargs) -> Dict[str, Any]:
        """Make API request with full enterprise features"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(endpoint, kwargs)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Rate limiting
        allowed, remaining = self.rate_limiter.is_allowed(self.api_key)
        if not allowed:
            raise Exception("Rate limit exceeded")
        
        # Circuit breaker protection
        @self.circuit_breaker
        def _make_request():
            url = urljoin(self.base_url, endpoint.url)
            response = self.session.request(
                method=endpoint.method,
                url=url,
                timeout=endpoint.timeout,
                headers=endpoint.headers,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        
        try:
            result = _make_request()
            
            # Cache successful results
            self.cache.set(cache_key, result, endpoint.cache_ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def _generate_cache_key(self, endpoint: APIEndpoint, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for request"""
        key_data = {
            "url": endpoint.url,
            "method": endpoint.method,
            "params": kwargs.get("params", {}),
            "data": kwargs.get("json", {})
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    async def batch_request(self, endpoints: List[APIEndpoint], **kwargs) -> List[Dict[str, Any]]:
        """Async batch processing for multiple API calls"""
        async def fetch_single(endpoint):
            return self.make_request(endpoint, **kwargs)
        
        tasks = [fetch_single(endpoint) for endpoint in endpoints]
        return await asyncio.gather(*tasks, return_exceptions=True)

class APIManager:
    """Central API management system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.webhook_handler = None
        self.api_clients = {}
        self.monitoring_stats = defaultdict(int)
    
    def register_api_client(self, name: str, base_url: str, api_key: str) -> EnterpriseAPIClient:
        """Register new API client"""
        client = EnterpriseAPIClient(base_url, api_key, self.redis_client)
        self.api_clients[name] = client
        return client
    
    def setup_webhook_server(self, secret_key: str, host: str = "0.0.0.0", port: int = 5000):
        """Setup webhook server"""
        self.webhook_handler = WebhookHandler(secret_key, self.redis_client)
        self.webhook_handler.app.run(host=host, port=port, threaded=True)
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics"""
        stats = dict(self.monitoring_stats)
        
        # Add cache statistics
        for name, client in self.api_clients.items():
            stats[f"{name}_cache_stats"] = client.cache.cache_stats
        
        # Add Redis statistics
        redis_info = self.redis_client.info()
        stats["redis_memory_usage"] = redis_info.get("used_memory_human")
        stats["redis_connected_clients"] = redis_info.get("connected_clients")
        
        return stats

# Global API manager instance
api_manager = APIManager()

# Decorator for easy API endpoint definition
def api_endpoint(url: str, method: str = "GET", cache_ttl: int = 300, rate_limit: int = 100):
    """Decorator for defining API endpoints"""
    def decorator(func: Callable) -> Callable:
        endpoint_config = APIEndpoint(
            url=url,
            method=method,
            cache_ttl=cache_ttl,
            rate_limit=rate_limit
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(endpoint_config, *args, **kwargs)
        
        return wrapper
    return decorator

# Usage example functions
@api_endpoint("/api/v1/stocks/{symbol}", cache_ttl=60)
def get_stock_data(endpoint: APIEndpoint, symbol: str, client_name: str = "default"):
    """Get stock data with enterprise features"""
    client = api_manager.api_clients.get(client_name)
    if not client:
        raise ValueError(f"API client '{client_name}' not found")
    
    endpoint.url = endpoint.url.format(symbol=symbol)
    return client.make_request(endpoint)

@api_endpoint("/api/v1/crypto/{symbol}", cache_ttl=30)
def get_crypto_data(endpoint: APIEndpoint, symbol: str, client_name: str = "crypto"):
    """Get cryptocurrency data"""
    client = api_manager.api_clients.get(client_name)
    if not client:
        raise ValueError(f"API client '{client_name}' not found")
    
    endpoint.url = endpoint.url.format(symbol=symbol)
    return client.make_request(endpoint) 