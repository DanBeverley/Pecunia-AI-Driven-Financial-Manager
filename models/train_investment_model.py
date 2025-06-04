import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping implementation for training"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model=None):
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best_weights:
                self.best_weights = model
        elif score > self.best_score - self.min_delta:  # For regression, higher is better (R²)
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = model
        return False

class InvestmentPredictor:
    def __init__(self, n_epochs=50, early_stopping_patience=10, regularization_strength=0.1, 
                 prediction_days=30):
        self.model = None
        self.scaler = StandardScaler()
        self.stock_encoder = LabelEncoder()
        self.feature_names = []
        self.model_metrics = {}
        self.training_history = {
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': [],
            'epoch_times': []
        }
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.regularization_strength = regularization_strength
        self.prediction_days = prediction_days
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for stock price prediction
        """
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['Name', 'date'])
        
        features_list = []
        
        for stock_name in df['Name'].unique():
            stock_data = df[df['Name'] == stock_name].copy()
            stock_data = stock_data.sort_values('date')
            
            # Price-based features
            stock_data['price_range'] = stock_data['high'] - stock_data['low']
            stock_data['price_change'] = stock_data['close'] - stock_data['open']
            stock_data['price_change_pct'] = stock_data['price_change'] / stock_data['open']
            
            # Technical indicators
            for window in [5, 10, 20, 50]:
                stock_data[f'sma_{window}'] = stock_data['close'].rolling(window=window).mean()
                stock_data[f'ema_{window}'] = stock_data['close'].ewm(span=window).mean()
                stock_data[f'volatility_{window}'] = stock_data['close'].rolling(window=window).std()
                stock_data[f'volume_sma_{window}'] = stock_data['volume'].rolling(window=window).mean()
            
            # RSI (Relative Strength Index)
            delta = stock_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            stock_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = stock_data['close'].ewm(span=12).mean()
            exp2 = stock_data['close'].ewm(span=26).mean()
            stock_data['macd'] = exp1 - exp2
            stock_data['macd_signal'] = stock_data['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            stock_data['bb_middle'] = stock_data['close'].rolling(window=20).mean()
            bb_std = stock_data['close'].rolling(window=20).std()
            stock_data['bb_upper'] = stock_data['bb_middle'] + (bb_std * 2)
            stock_data['bb_lower'] = stock_data['bb_middle'] - (bb_std * 2)
            stock_data['bb_position'] = (stock_data['close'] - stock_data['bb_lower']) / (stock_data['bb_upper'] - stock_data['bb_lower'])
            
            # Lagged features
            for lag in [1, 3, 5, 10]:
                stock_data[f'close_lag_{lag}'] = stock_data['close'].shift(lag)
                stock_data[f'volume_lag_{lag}'] = stock_data['volume'].shift(lag)
                stock_data[f'price_change_lag_{lag}'] = stock_data['price_change'].shift(lag)
            
            # Time-based features
            stock_data['day_of_week'] = stock_data['date'].dt.dayofweek
            stock_data['month'] = stock_data['date'].dt.month
            stock_data['quarter'] = stock_data['date'].dt.quarter
            stock_data['is_month_end'] = (stock_data['date'].dt.day > 25).astype(int)
            
            # Target: Future price (next N days)
            stock_data['target'] = stock_data['close'].shift(-self.prediction_days)
            stock_data['target_return'] = (stock_data['target'] - stock_data['close']) / stock_data['close']
            
            features_list.append(stock_data)
        
        return pd.concat(features_list, ignore_index=True)
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Advanced preprocessing for investment prediction
        """
        print("🔧 Engineering features...")
        df = self.engineer_features(data)
        
        # Remove rows with NaN values (due to rolling windows and lags)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No data remaining after feature engineering and NaN removal")
        
        # Select features for training
        feature_columns = [
            'open', 'high', 'low', 'volume', 'price_range', 'price_change', 'price_change_pct',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
            'volume_sma_5', 'volume_sma_10', 'volume_sma_20', 'volume_sma_50',
            'rsi', 'macd', 'macd_signal', 'bb_position',
            'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_10',
            'volume_lag_1', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
            'price_change_lag_1', 'price_change_lag_3', 'price_change_lag_5', 'price_change_lag_10',
            'day_of_week', 'month', 'quarter', 'is_month_end'
        ]
        
        # Add stock encoding
        df['stock_encoded'] = self.stock_encoder.fit_transform(df['Name'])
        feature_columns.append('stock_encoded')
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['target'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        self.feature_names = feature_columns
        
        print(f"✅ Features engineered: {len(feature_columns)} features, {len(X_scaled)} samples")
        
        return X_scaled, y
    
    def create_ensemble_model(self) -> Pipeline:
        """
        Create advanced ensemble regressor with regularization
        """
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.n_epochs,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=self.regularization_strength,  # L1 regularization
            reg_lambda=self.regularization_strength,  # L2 regularization
            random_state=42,
            n_jobs=-1
        )
        
        rf_model = RandomForestRegressor(
            n_estimators=self.n_epochs,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Regularization through feature subsampling
            random_state=42,
            n_jobs=-1
        )
        
        # Voting ensemble
        ensemble = VotingRegressor([
            ('xgb', xgb_model),
            ('rf', rf_model)
        ])
        
        pipeline = Pipeline([
            ('regressor', ensemble)
        ])
        
        return pipeline
    
    def train_model_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train ensemble model with time series cross-validation and progress tracking
        """
        print(f"🚀 Starting investment model training with {self.n_epochs} epochs...")
        
        # Time series cross-validation (respects temporal order)
        cv_folds = 5
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_scores = []
        
        print(f"📊 Performing {cv_folds}-fold time series cross-validation...")
        
        # Cross-validation with progress bar
        cv_progress = tqdm(tscv.split(X), total=cv_folds, desc="CV Folds")
        
        for fold, (train_idx, val_idx) in enumerate(cv_progress):
            # Convert to DataFrame if it's numpy array
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
                X_train_fold, X_val_fold = X_df.iloc[train_idx], X_df.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model for this fold
            fold_model = self.create_ensemble_model()
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate fold
            val_pred = fold_model.predict(X_val_fold)
            fold_score = r2_score(y_val_fold, val_pred)
            cv_scores.append(fold_score)
            
            cv_progress.set_postfix({
                'Fold': fold + 1,
                'R²': f'{fold_score:.4f}',
                'Mean': f'{np.mean(cv_scores):.4f}'
            })
        
        cv_progress.close()
        
        # Final train-test split (time-aware)
        split_index = int(len(X) * 0.8)  # Use last 20% for testing
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        # Hyperparameter grid with regularization focus
        param_grid = {
            'regressor__xgb__n_estimators': [50, 100, 150],
            'regressor__xgb__max_depth': [4, 6, 8],
            'regressor__xgb__learning_rate': [0.05, 0.1, 0.15],
            'regressor__xgb__reg_alpha': [0.01, 0.1, 0.2],
            'regressor__xgb__reg_lambda': [0.01, 0.1, 0.2],
            'regressor__rf__n_estimators': [50, 100],
            'regressor__rf__max_depth': [10, 12, 15],
            'regressor__rf__min_samples_split': [5, 10, 15]
        }
        
        base_model = self.create_ensemble_model()
        
        print("🔍 Performing hyperparameter optimization...")
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=TimeSeriesSplit(n_splits=3), scoring='r2',
            n_jobs=-1, verbose=1
        )
        
        # Training with progress tracking
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.model = grid_search.best_estimator_
        
        # Evaluate model performance
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate comprehensive metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
        test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
        
        self.model_metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'best_params': grid_search.best_params_,
            'grid_search_score': grid_search.best_score_,
            'training_time': training_time,
            'n_epochs': self.n_epochs,
            'regularization_strength': self.regularization_strength,
            'prediction_days': self.prediction_days
        }
        
        # Store training history
        self.training_history['train_rmse'].append(train_rmse)
        self.training_history['val_rmse'].append(test_rmse)
        self.training_history['train_r2'].append(train_r2)
        self.training_history['val_r2'].append(test_r2)
        self.training_history['epoch_times'].append(training_time)
        
        logger.info(f"✅ Investment model trained successfully!")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Test RMSE: ${test_rmse:.2f}")
        logger.info(f"CV Mean: {self.model_metrics['cv_mean']:.4f} ± {self.model_metrics['cv_std']:.4f}")
        
        return self.model
    
    def plot_training_history(self, save_path: str = 'investment_training_history.png'):
        """Plot training history"""
        if not self.training_history['train_rmse']:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # RMSE plot
        ax1.plot(self.training_history['train_rmse'], label='Train RMSE', marker='o')
        ax1.plot(self.training_history['val_rmse'], label='Validation RMSE', marker='s')
        ax1.set_title('Investment Model RMSE')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE ($)')
        ax1.legend()
        ax1.grid(True)
        
        # R² plot
        ax2.plot(self.training_history['train_r2'], label='Train R²', marker='o')
        ax2.plot(self.training_history['val_r2'], label='Validation R²', marker='s')
        ax2.set_title('Investment Model R² Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Investment training history saved to {save_path}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make price predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Use the same preprocessing pipeline
        features, _ = self.preprocess_data(data)
        
        return self.model.predict(features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        if self.model is None:
            return {}
        
        try:
            # Extract feature importance from ensemble components
            xgb_regressor = self.model.named_steps['regressor'].estimators[0]
            rf_regressor = self.model.named_steps['regressor'].estimators[1]
            
            # Average importance from both models
            xgb_importance = xgb_regressor.feature_importances_
            rf_importance = rf_regressor.feature_importances_
            
            avg_importance = (xgb_importance + rf_importance) / 2
            
            importance_dict = dict(zip(self.feature_names, avg_importance))
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str = 'investment_predictor_model.pkl') -> bool:
        """Save the trained model and preprocessors"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'stock_encoder': self.stock_encoder,
                'feature_names': self.feature_names,
                'metrics': self.model_metrics,
                'training_history': self.training_history,
                'hyperparameters': {
                    'n_epochs': self.n_epochs,
                    'early_stopping_patience': self.early_stopping_patience,
                    'regularization_strength': self.regularization_strength,
                    'prediction_days': self.prediction_days
                },
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str = 'investment_predictor_model.pkl') -> bool:
        """Load a trained model and preprocessors"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', StandardScaler())
            self.stock_encoder = model_data.get('stock_encoder', LabelEncoder())
            self.feature_names = model_data.get('feature_names', [])
            self.model_metrics = model_data.get('metrics', {})
            self.training_history = model_data.get('training_history', {})
            
            # Load hyperparameters if available
            hyperparams = model_data.get('hyperparameters', {})
            self.n_epochs = hyperparams.get('n_epochs', 50)
            self.early_stopping_patience = hyperparams.get('early_stopping_patience', 10)
            self.regularization_strength = hyperparams.get('regularization_strength', 0.1)
            self.prediction_days = hyperparams.get('prediction_days', 30)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def train_investment_prediction_model(data_path: str = '../all_stocks_5yr.csv',
                                    n_epochs: int = 50,
                                    early_stopping_patience: int = 10,
                                    regularization_strength: float = 0.1,
                                    prediction_days: int = 30,
                                    sample_size: int = 50000,  # Limit for faster training
                                    save_model: bool = True,
                                    model_path: str = 'investment_predictor.pkl') -> InvestmentPredictor:
    """
    Train and return an investment prediction model using stock data with advanced features
    """
    # Load stock dataset
    try:
        data = pd.read_csv(data_path)
        print(f"📈 Loaded stock dataset with {len(data)} records and {len(data.columns)} features")
        print(f"🏢 Unique stocks: {data['Name'].nunique()}")
        print(f"📅 Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Sample data for faster training if specified
        if sample_size and sample_size < len(data):
            data = data.sample(n=sample_size, random_state=42)
            print(f"🎯 Using sample of {len(data)} records for training")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    predictor = InvestmentPredictor(
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        regularization_strength=regularization_strength,
        prediction_days=prediction_days
    )
    
    # Preprocess data
    print("🔄 Preprocessing stock data...")
    X, y = predictor.preprocess_data(data)
    
    # Train model with advanced features
    predictor.train_model_with_cv(X, y)
    
    # Save model if requested
    if save_model:
        predictor.save_model(model_path)
    
    # Print comprehensive results
    metrics = predictor.model_metrics
    print(f"\n{'='*50}")
    print(f"📈 INVESTMENT PREDICTION MODEL RESULTS")
    print(f"{'='*50}")
    print(f"📊 Performance Metrics:")
    print(f"   • Test R² Score: {metrics['test_r2']:.4f}")
    print(f"   • Test RMSE: ${metrics['test_rmse']:.2f}")
    print(f"   • Test MAE: ${metrics['test_mae']:.2f}")
    print(f"   • Test MAPE: {metrics['test_mape']:.2%}")
    print(f"   • Training Time: {metrics['training_time']:.2f}s")
    print(f"\n🔀 Cross-Validation:")
    print(f"   • CV Mean R²: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"   • All CV Scores: {[f'{s:.3f}' for s in metrics['cv_scores']]}")
    print(f"\n⚙️  Training Configuration:")
    print(f"   • Epochs: {metrics['n_epochs']}")
    print(f"   • Regularization: {metrics['regularization_strength']}")
    print(f"   • Prediction Days: {metrics['prediction_days']}")
    print(f"   • Early Stopping: {early_stopping_patience} patience")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\n🔍 Top 15 Important Features:")
    for i, (feature, score) in enumerate(list(importance.items())[:15]):
        print(f"   {i+1:2d}. {feature:<20}: {score:.4f}")
    
    # Plot training history
    predictor.plot_training_history()
    
    return predictor

# Backward compatibility functions
def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Legacy function for backward compatibility"""
    predictor = InvestmentPredictor()
    return predictor.preprocess_data(data)

def train_model(X: pd.DataFrame, y: pd.Series) -> Any:
    """Legacy function for backward compatibility"""
    predictor = InvestmentPredictor()
    return predictor.train_model_with_cv(X, y)

if __name__ == "__main__":
    print("📈 Training investment prediction model on stock dataset...")
    predictor = train_investment_prediction_model(
        sample_size=50000,  # Use 50k samples for demo
        n_epochs=50,
        early_stopping_patience=10,
        regularization_strength=0.1,
        prediction_days=30
    )
    
    if predictor:
        print(f"\n✅ Investment model training completed successfully!")
        print(f"📁 Model saved as: investment_predictor.pkl")
        print(f"📈 Training history plot saved as: investment_training_history.png")
        print(f"🎯 Model predicts stock prices {predictor.prediction_days} days into the future") 