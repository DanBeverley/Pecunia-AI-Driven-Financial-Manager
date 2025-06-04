"""
Pecunia AI - Advanced Data Processing & Anomaly Detection System
Streaming data processing, ML-based anomaly detection, and intelligent data cleaning
"""

import asyncio
import json
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Iterator
import warnings
warnings.filterwarnings('ignore')

# Scientific computing and ML
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import silhouette_score
import joblib

# Time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Stream processing
import redis
from kafka import KafkaConsumer, KafkaProducer
import asyncpg

# Data validation
from pydantic import BaseModel, validator
from marshmallow import Schema, fields, validate

# Configure advanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class AnomalyType(Enum):
    """Types of anomalies detected"""
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    BEHAVIORAL = "behavioral"

class ProcessingMode(Enum):
    """Data processing modes"""
    BATCH = "batch"
    STREAM = "stream"
    REAL_TIME = "real_time"
    HYBRID = "hybrid"

@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics"""
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0
    timeliness: float = 0.0
    overall_score: float = 0.0
    anomaly_count: int = 0
    data_drift_score: float = 0.0

@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection algorithms"""
    isolation_forest_contamination: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    statistical_threshold: float = 3.0
    ensemble_voting_threshold: float = 0.6
    temporal_window_size: int = 100
    enable_adaptive_thresholds: bool = True

class DataValidator(BaseModel):
    """Pydantic model for financial data validation"""
    timestamp: datetime
    amount: float
    category: Optional[str] = None
    user_id: str
    transaction_type: str
    
    @validator('amount')
    def validate_amount(cls, v):
        if v < 0 and v < -1000000:  # Reasonable limit for negative amounts
            raise ValueError('Amount is suspiciously large negative value')
        if v > 1000000000:  # 1 billion limit
            raise ValueError('Amount exceeds reasonable limit')
        return v
    
    @validator('transaction_type')
    def validate_transaction_type(cls, v):
        valid_types = {'debit', 'credit', 'transfer', 'payment', 'refund'}
        if v.lower() not in valid_types:
            raise ValueError(f'Invalid transaction type: {v}')
        return v.lower()

class StatisticalAnomalyDetector:
    """Advanced statistical anomaly detection"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.z_threshold = config.statistical_threshold
        self.iqr_multiplier = 1.5
        self.adaptive_thresholds = {}
        
    def detect_outliers_zscore(self, data: np.ndarray, adaptive: bool = True) -> np.ndarray:
        """Z-score based outlier detection with adaptive thresholds"""
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
        
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        
        if adaptive and len(data) > 100:
            # Adaptive threshold based on data distribution
            threshold = self._calculate_adaptive_threshold(data)
        else:
            threshold = self.z_threshold
            
        return z_scores > threshold
    
    def detect_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """IQR-based outlier detection"""
        if len(data) < 4:
            return np.zeros(len(data), dtype=bool)
            
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.iqr_multiplier * IQR
        upper_bound = Q3 + self.iqr_multiplier * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def detect_outliers_modified_zscore(self, data: np.ndarray) -> np.ndarray:
        """Modified Z-score using median absolute deviation"""
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
            
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return np.zeros(len(data), dtype=bool)
            
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > 3.5
    
    def _calculate_adaptive_threshold(self, data: np.ndarray) -> float:
        """Calculate adaptive threshold based on data characteristics"""
        # Consider data distribution properties
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Adjust threshold based on distribution characteristics
        base_threshold = self.z_threshold
        
        if abs(skewness) > 1:  # Highly skewed data
            base_threshold *= 1.2
        if kurtosis > 3:  # Heavy-tailed distribution
            base_threshold *= 1.1
            
        return base_threshold

class MLAnomalyDetector:
    """Machine learning based anomaly detection"""
    
    def __init__(self, config: AnomalyDetectionConfig):
        self.config = config
        self.isolation_forest = IsolationForest(
            contamination=config.isolation_forest_contamination,
            random_state=42,
            n_estimators=100
        )
        self.dbscan = DBSCAN(
            eps=config.dbscan_eps,
            min_samples=config.dbscan_min_samples
        )
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit anomaly detection models"""
        if len(X) < 50:
            logger.warning("Insufficient data for ML anomaly detection")
            return
            
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
        
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies using ensemble of ML methods"""
        if not self.is_fitted:
            logger.warning("Models not fitted, fitting on current data")
            self.fit(X)
            
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest
        iso_anomalies = self.isolation_forest.predict(X_scaled) == -1
        
        # DBSCAN clustering
        cluster_labels = self.dbscan.fit_predict(X_scaled)
        dbscan_anomalies = cluster_labels == -1
        
        # Ensemble voting
        ensemble_score = (iso_anomalies.astype(int) + dbscan_anomalies.astype(int)) / 2
        return ensemble_score >= self.config.ensemble_voting_threshold

class TimeSeriesAnomalyDetector:
    """Specialized time series anomaly detection"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.seasonal_decomposition_cache = {}
        
    def detect_seasonal_anomalies(self, data: pd.Series, freq: int = None) -> np.ndarray:
        """Detect anomalies in seasonal patterns"""
        if len(data) < 2 * self.window_size:
            return np.zeros(len(data), dtype=bool)
        
        try:
            # Perform seasonal decomposition
            if freq is None:
                freq = min(len(data) // 4, 30)  # Auto-detect frequency
                
            decomposition = seasonal_decompose(
                data, 
                model='additive', 
                period=freq,
                extrapolate_trend='freq'
            )
            
            # Detect anomalies in residuals
            residuals = decomposition.resid.dropna()
            threshold = 3 * residuals.std()
            
            anomalies = np.zeros(len(data), dtype=bool)
            residual_anomalies = np.abs(residuals) > threshold
            
            # Map back to original data indices
            valid_indices = residuals.index
            anomalies[valid_indices] = residual_anomalies.values
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Seasonal anomaly detection failed: {e}")
            return np.zeros(len(data), dtype=bool)
    
    def detect_trend_anomalies(self, data: pd.Series) -> np.ndarray:
        """Detect anomalies in trend patterns"""
        if len(data) < 20:
            return np.zeros(len(data), dtype=bool)
            
        # Calculate rolling trend
        rolling_mean = data.rolling(window=min(len(data)//4, 20)).mean()
        rolling_std = data.rolling(window=min(len(data)//4, 20)).std()
        
        # Detect points that deviate significantly from trend
        deviations = np.abs(data - rolling_mean) / (rolling_std + 1e-8)
        threshold = 2.5
        
        return deviations > threshold
    
    def detect_change_points(self, data: np.ndarray) -> List[int]:
        """Detect change points in time series"""
        if len(data) < 20:
            return []
            
        # Use PELT algorithm approximation
        change_points = []
        window_size = min(len(data) // 10, 50)
        
        for i in range(window_size, len(data) - window_size):
            before = data[i-window_size:i]
            after = data[i:i+window_size]
            
            # Statistical test for change point
            statistic, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # Significant change
                change_points.append(i)
                
        return change_points

class DataQualityAssessor:
    """Comprehensive data quality assessment"""
    
    def __init__(self):
        self.quality_thresholds = {
            'completeness': {'excellent': 0.95, 'good': 0.90, 'fair': 0.80, 'poor': 0.70},
            'accuracy': {'excellent': 0.95, 'good': 0.90, 'fair': 0.85, 'poor': 0.75},
            'consistency': {'excellent': 0.98, 'good': 0.95, 'fair': 0.90, 'poor': 0.80},
        }
    
    def assess_completeness(self, df: pd.DataFrame) -> float:
        """Assess data completeness"""
        if df.empty:
            return 0.0
            
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        return completeness
    
    def assess_accuracy(self, df: pd.DataFrame, validation_rules: Dict[str, Callable] = None) -> float:
        """Assess data accuracy using validation rules"""
        if df.empty:
            return 0.0
            
        total_records = len(df)
        accurate_records = 0
        
        if validation_rules:
            for column, rule in validation_rules.items():
                if column in df.columns:
                    try:
                        valid_mask = df[column].apply(rule)
                        accurate_records += valid_mask.sum()
                    except Exception as e:
                        logger.warning(f"Validation rule failed for {column}: {e}")
        else:
            # Default accuracy assessment
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                # Check for reasonable ranges (no extreme outliers)
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                reasonable_range = (df[col] >= q1 - 3*iqr) & (df[col] <= q3 + 3*iqr)
                accurate_records += reasonable_range.sum()
        
        return accurate_records / (total_records * len(df.columns)) if total_records > 0 else 0.0
    
    def assess_consistency(self, df: pd.DataFrame) -> float:
        """Assess data consistency"""
        if df.empty:
            return 0.0
            
        consistency_score = 1.0
        
        # Check for duplicate records
        if len(df) > 1:
            duplicate_ratio = df.duplicated().sum() / len(df)
            consistency_score -= duplicate_ratio * 0.3
        
        # Check for consistent data types within columns
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check for consistent formatting
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    # Simple consistency check for string columns
                    patterns = non_null_values.astype(str).str.len().nunique()
                    if patterns > len(non_null_values) * 0.5:  # Too many different patterns
                        consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def calculate_overall_quality(self, metrics: DataQualityMetrics) -> DataQuality:
        """Calculate overall data quality level"""
        scores = [
            metrics.completeness,
            metrics.accuracy,
            metrics.consistency,
            metrics.validity,
            metrics.uniqueness,
            metrics.timeliness
        ]
        
        overall_score = np.mean([s for s in scores if s > 0])
        
        if overall_score >= 0.90:
            return DataQuality.EXCELLENT
        elif overall_score >= 0.80:
            return DataQuality.GOOD
        elif overall_score >= 0.70:
            return DataQuality.FAIR
        elif overall_score >= 0.60:
            return DataQuality.POOR
        else:
            return DataQuality.CRITICAL

class StreamProcessor:
    """High-performance stream processing system"""
    
    def __init__(self, redis_client: redis.Redis, kafka_config: Dict[str, Any] = None):
        self.redis = redis_client
        self.kafka_config = kafka_config or {}
        self.processing_stats = defaultdict(int)
        self.buffers = defaultdict(deque)
        self.processors = {}
        
    async def process_stream(self, stream_name: str, processor_func: Callable, 
                           batch_size: int = 100, max_wait_ms: int = 1000):
        """Process streaming data with batching"""
        consumer = self._create_kafka_consumer(stream_name) if self.kafka_config else None
        
        while True:
            try:
                batch = []
                start_time = time.time()
                
                # Collect batch from stream
                if consumer:
                    # Kafka stream processing
                    for message in consumer:
                        batch.append(json.loads(message.value.decode('utf-8')))
                        
                        if len(batch) >= batch_size or \
                           (time.time() - start_time) * 1000 >= max_wait_ms:
                            break
                else:
                    # Redis stream processing
                    batch = await self._read_redis_stream(stream_name, batch_size, max_wait_ms)
                
                if batch:
                    # Process batch
                    try:
                        results = await self._process_batch(batch, processor_func)
                        self.processing_stats[f"{stream_name}_processed"] += len(batch)
                        
                        # Store results
                        await self._store_processed_results(stream_name, results)
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed for {stream_name}: {e}")
                        self.processing_stats[f"{stream_name}_errors"] += 1
                        
                await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Stream processing error for {stream_name}: {e}")
                await asyncio.sleep(1)  # Longer delay on error
    
    async def _read_redis_stream(self, stream_name: str, batch_size: int, max_wait_ms: int) -> List[Dict]:
        """Read batch from Redis stream"""
        try:
            # Use Redis XREAD with block and count
            result = self.redis.xread({stream_name: '$'}, count=batch_size, block=max_wait_ms)
            
            batch = []
            for stream, messages in result:
                for message_id, fields in messages:
                    data = {k.decode(): v.decode() for k, v in fields.items()}
                    batch.append(data)
                    
            return batch
            
        except Exception as e:
            logger.error(f"Redis stream read error: {e}")
            return []
    
    async def _process_batch(self, batch: List[Dict], processor_func: Callable) -> List[Dict]:
        """Process batch of data"""
        if asyncio.iscoroutinefunction(processor_func):
            return await processor_func(batch)
        else:
            # Run in thread pool for CPU-bound processing
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, processor_func, batch)
                return await future
    
    async def _store_processed_results(self, stream_name: str, results: List[Dict]):
        """Store processed results"""
        result_key = f"processed:{stream_name}"
        
        for result in results:
            self.redis.lpush(result_key, json.dumps(result))
            
        # Keep only recent results
        self.redis.ltrim(result_key, 0, 10000)
        self.redis.expire(result_key, 86400)  # 24 hours
    
    def _create_kafka_consumer(self, topic: str) -> KafkaConsumer:
        """Create Kafka consumer"""
        return KafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_config.get('bootstrap_servers', ['localhost:9092']),
            value_deserializer=lambda x: x,
            consumer_timeout_ms=1000
        )

class AdvancedDataCleaner:
    """Enterprise-grade data cleaning system"""
    
    def __init__(self, anomaly_config: AnomalyDetectionConfig = None):
        self.anomaly_config = anomaly_config or AnomalyDetectionConfig()
        self.statistical_detector = StatisticalAnomalyDetector(self.anomaly_config)
        self.ml_detector = MLAnomalyDetector(self.anomaly_config)
        self.ts_detector = TimeSeriesAnomalyDetector()
        self.quality_assessor = DataQualityAssessor()
        self.cleaning_stats = defaultdict(int)
        
    def clean_financial_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Comprehensive financial data cleaning"""
        logger.info(f"Starting data cleaning for {len(df)} records")
        original_shape = df.shape
        
        # Step 1: Basic cleaning
        df_cleaned = self._basic_cleaning(df.copy())
        
        # Step 2: Handle missing values intelligently
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Step 3: Detect and handle outliers
        df_cleaned = self._handle_outliers(df_cleaned, target_column)
        
        # Step 4: Standardize formats
        df_cleaned = self._standardize_formats(df_cleaned)
        
        # Step 5: Feature engineering for quality improvement
        df_cleaned = self._engineer_quality_features(df_cleaned)
        
        # Step 6: Validate data integrity
        df_cleaned = self._validate_data_integrity(df_cleaned)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(df_cleaned, original_shape)
        
        logger.info(f"Data cleaning completed. Shape: {original_shape} -> {df_cleaned.shape}")
        logger.info(f"Quality score: {quality_metrics.overall_score:.3f}")
        
        return df_cleaned, quality_metrics
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning operations"""
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Remove duplicate rows
        initial_len = len(df)
        df = df.drop_duplicates()
        self.cleaning_stats['duplicates_removed'] = initial_len - len(df)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Remove rows with all zeros (likely placeholder data)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            zero_mask = (df[numeric_columns] == 0).all(axis=1)
            df = df[~zero_mask]
            self.cleaning_stats['zero_rows_removed'] = zero_mask.sum()
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent missing value handling"""
        for column in df.columns:
            missing_ratio = df[column].isnull().sum() / len(df)
            
            if missing_ratio > 0.8:
                # Drop columns with too many missing values
                df = df.drop(columns=[column])
                self.cleaning_stats[f'{column}_dropped_high_missing'] = 1
                continue
            
            if df[column].dtype in ['int64', 'float64']:
                # Numeric columns: use advanced imputation
                if missing_ratio < 0.1:
                    # Forward fill for time series-like data
                    df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
                elif missing_ratio < 0.3:
                    # Use median for moderate missing data
                    df[column] = df[column].fillna(df[column].median())
                else:
                    # Use interpolation for high missing data
                    df[column] = df[column].interpolate(method='linear')
                    
            elif df[column].dtype == 'object':
                # Categorical columns
                mode_value = df[column].mode()
                if len(mode_value) > 0:
                    df[column] = df[column].fillna(mode_value.iloc[0])
                else:
                    df[column] = df[column].fillna('unknown')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """Advanced outlier detection and handling"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if len(df[column].dropna()) < 10:
                continue
                
            data = df[column].dropna().values
            
            # Use ensemble of outlier detection methods
            outliers_z = self.statistical_detector.detect_outliers_zscore(data)
            outliers_iqr = self.statistical_detector.detect_outliers_iqr(data)
            outliers_modified_z = self.statistical_detector.detect_outliers_modified_zscore(data)
            
            # Ensemble voting
            outlier_votes = outliers_z.astype(int) + outliers_iqr.astype(int) + outliers_modified_z.astype(int)
            consensus_outliers = outlier_votes >= 2  # At least 2 methods agree
            
            if consensus_outliers.any():
                # Handle outliers based on column importance
                if column == target_column:
                    # For target column, cap outliers instead of removing
                    q1, q99 = np.percentile(data, [1, 99])
                    data_capped = np.clip(data, q1, q99)
                    df.loc[df[column].notna(), column] = data_capped
                else:
                    # For other columns, remove extreme outliers
                    valid_indices = df[column].notna()
                    extreme_outliers = consensus_outliers & (outlier_votes == 3)  # All methods agree
                    
                    if extreme_outliers.any():
                        outlier_indices = df.index[valid_indices][extreme_outliers]
                        df = df.drop(outlier_indices)
                        self.cleaning_stats[f'{column}_outliers_removed'] = len(outlier_indices)
        
        return df
    
    def _standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data formats"""
        for column in df.columns:
            if df[column].dtype == 'object':
                # String standardization
                df[column] = df[column].astype(str).str.strip().str.lower()
                
                # Detect and convert date columns
                if self._is_date_column(df[column]):
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    except Exception:
                        pass
                
                # Detect and convert numeric columns stored as strings
                elif self._is_numeric_string(df[column]):
                    try:
                        # Remove common non-numeric characters
                        cleaned = df[column].str.replace('[,$%]', '', regex=True)
                        df[column] = pd.to_numeric(cleaned, errors='coerce')
                    except Exception:
                        pass
        
        return df
    
    def _engineer_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features to improve data quality"""
        # Add data quality indicators
        df['_record_completeness'] = (df.notna().sum(axis=1) / len(df.columns))
        
        # Add time-based features if datetime column exists
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            dt_col = datetime_columns[0]
            df['_hour'] = df[dt_col].dt.hour
            df['_day_of_week'] = df[dt_col].dt.dayofweek
            df['_is_weekend'] = df['_day_of_week'].isin([5, 6])
        
        # Add derived financial features
        if 'amount' in df.columns:
            df['_amount_abs'] = df['amount'].abs()
            df['_is_large_transaction'] = df['_amount_abs'] > df['_amount_abs'].quantile(0.95)
            
            # Rolling statistics
            if len(df) > 10:
                df['_amount_rolling_mean'] = df['amount'].rolling(window=10, min_periods=1).mean()
                df['_amount_rolling_std'] = df['amount'].rolling(window=10, min_periods=1).std()
        
        return df
    
    def _validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data integrity using business rules"""
        # Financial data specific validations
        if 'amount' in df.columns and 'transaction_type' in df.columns:
            # Ensure credit amounts are positive and debit amounts are negative
            credit_mask = df['transaction_type'].str.contains('credit', na=False)
            debit_mask = df['transaction_type'].str.contains('debit', na=False)
            
            # Fix sign inconsistencies
            df.loc[credit_mask & (df['amount'] < 0), 'amount'] *= -1
            df.loc[debit_mask & (df['amount'] > 0), 'amount'] *= -1
        
        # Remove records with impossible dates
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        for dt_col in datetime_columns:
            future_mask = df[dt_col] > datetime.now()
            ancient_mask = df[dt_col] < datetime(1900, 1, 1)
            invalid_dates = future_mask | ancient_mask
            
            if invalid_dates.any():
                df = df[~invalid_dates]
                self.cleaning_stats[f'{dt_col}_invalid_dates_removed'] = invalid_dates.sum()
        
        return df
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, original_shape: Tuple[int, int]) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics"""
        metrics = DataQualityMetrics()
        
        # Completeness
        metrics.completeness = self.quality_assessor.assess_completeness(df)
        
        # Accuracy (based on validation rules)
        metrics.accuracy = self.quality_assessor.assess_accuracy(df)
        
        # Consistency
        metrics.consistency = self.quality_assessor.assess_consistency(df)
        
        # Validity (proportion of valid records)
        metrics.validity = len(df) / original_shape[0] if original_shape[0] > 0 else 0.0
        
        # Uniqueness (proportion of unique records)
        metrics.uniqueness = 1 - (df.duplicated().sum() / len(df)) if len(df) > 0 else 0.0
        
        # Timeliness (based on most recent data)
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            most_recent = df[datetime_columns[0]].max()
            days_old = (datetime.now() - most_recent).days if pd.notna(most_recent) else 365
            metrics.timeliness = max(0, 1 - (days_old / 30))  # Decay over 30 days
        else:
            metrics.timeliness = 1.0  # Assume timely if no date column
        
        # Overall score
        scores = [metrics.completeness, metrics.accuracy, metrics.consistency, 
                 metrics.validity, metrics.uniqueness, metrics.timeliness]
        metrics.overall_score = np.mean([s for s in scores if s > 0])
        
        return metrics
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Detect if a column contains dates"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
            
        date_indicators = ['date', 'time', 'created', 'updated', 'timestamp']
        column_name = series.name.lower() if series.name else ''
        
        if any(indicator in column_name for indicator in date_indicators):
            return True
            
        # Try parsing a sample of values
        try:
            parsed_count = 0
            for value in sample.head(10):
                try:
                    pd.to_datetime(value)
                    parsed_count += 1
                except:
                    pass
            return parsed_count / len(sample.head(10)) > 0.5
        except:
            return False
    
    def _is_numeric_string(self, series: pd.Series) -> bool:
        """Detect if a string column contains numeric data"""
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
            
        try:
            # Remove common non-numeric characters and try conversion
            cleaned = sample.astype(str).str.replace('[,$%]', '', regex=True)
            numeric_count = pd.to_numeric(cleaned, errors='coerce').notna().sum()
            return numeric_count / len(sample) > 0.7
        except:
            return False

class RealTimeAnomalyMonitor:
    """Real-time anomaly monitoring system"""
    
    def __init__(self, redis_client: redis.Redis, alert_threshold: int = 10):
        self.redis = redis_client
        self.alert_threshold = alert_threshold
        self.anomaly_buffer = deque(maxlen=1000)
        self.alert_callbacks = []
        
    async def monitor_stream(self, stream_name: str, detector_func: Callable):
        """Monitor stream for anomalies in real-time"""
        while True:
            try:
                # Read latest data from stream
                data = await self._read_stream_data(stream_name)
                
                if data:
                    # Detect anomalies
                    anomalies = detector_func(data)
                    
                    if anomalies.any():
                        await self._handle_anomalies(stream_name, data, anomalies)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Anomaly monitoring error for {stream_name}: {e}")
                await asyncio.sleep(5)
    
    async def _read_stream_data(self, stream_name: str) -> Optional[np.ndarray]:
        """Read latest data from stream"""
        try:
            # Get recent data from Redis stream
            result = self.redis.xrevrange(stream_name, count=100)
            
            if result:
                data = []
                for stream_id, fields in result:
                    # Extract numeric value (assuming 'value' field)
                    if b'value' in fields:
                        try:
                            value = float(fields[b'value'])
                            data.append(value)
                        except ValueError:
                            pass
                
                return np.array(data) if data else None
                
        except Exception as e:
            logger.error(f"Stream data read error: {e}")
            
        return None
    
    async def _handle_anomalies(self, stream_name: str, data: np.ndarray, anomalies: np.ndarray):
        """Handle detected anomalies"""
        anomaly_count = anomalies.sum()
        self.anomaly_buffer.append({
            'timestamp': datetime.now(),
            'stream': stream_name,
            'anomaly_count': anomaly_count,
            'total_points': len(data)
        })
        
        # Check if alert threshold is exceeded
        recent_anomalies = sum(1 for entry in self.anomaly_buffer 
                             if entry['timestamp'] > datetime.now() - timedelta(minutes=5))
        
        if recent_anomalies >= self.alert_threshold:
            await self._trigger_alert(stream_name, recent_anomalies)
    
    async def _trigger_alert(self, stream_name: str, anomaly_count: int):
        """Trigger anomaly alert"""
        alert_data = {
            'stream': stream_name,
            'anomaly_count': anomaly_count,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if anomaly_count > self.alert_threshold * 2 else 'medium'
        }
        
        # Store alert in Redis
        self.redis.lpush('anomaly_alerts', json.dumps(alert_data))
        self.redis.expire('anomaly_alerts', 86400)  # Keep for 24 hours
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for anomaly alerts"""
        self.alert_callbacks.append(callback)

# Global instances
data_cleaner = AdvancedDataCleaner()
stream_processor = None
anomaly_monitor = None

def initialize_data_processing(redis_url: str = "redis://localhost:6379", 
                             kafka_config: Dict[str, Any] = None):
    """Initialize the data processing system"""
    global stream_processor, anomaly_monitor
    
    redis_client = redis.from_url(redis_url)
    stream_processor = StreamProcessor(redis_client, kafka_config)
    anomaly_monitor = RealTimeAnomalyMonitor(redis_client)
    
    return stream_processor, anomaly_monitor

# Utility functions for easy usage
def clean_data(df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, DataQualityMetrics]:
    """Quick data cleaning interface"""
    return data_cleaner.clean_financial_data(df, target_column)

def detect_anomalies(data: np.ndarray, method: str = "ensemble") -> np.ndarray:
    """Quick anomaly detection interface"""
    if method == "statistical":
        return data_cleaner.statistical_detector.detect_outliers_zscore(data)
    elif method == "ml":
        data_2d = data.reshape(-1, 1)
        data_cleaner.ml_detector.fit(data_2d)
        return data_cleaner.ml_detector.detect_anomalies(data_2d)
    elif method == "ensemble":
        # Use multiple methods and combine results
        z_outliers = data_cleaner.statistical_detector.detect_outliers_zscore(data)
        iqr_outliers = data_cleaner.statistical_detector.detect_outliers_iqr(data)
        return z_outliers | iqr_outliers
    else:
        raise ValueError(f"Unknown method: {method}") 