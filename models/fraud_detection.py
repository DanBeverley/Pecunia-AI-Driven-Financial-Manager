import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, accuracy_score, f1_score, precision_score, 
                           recall_score, roc_auc_score, confusion_matrix, precision_recall_curve,
                           average_precision_score)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
from datetime import datetime
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best_weights:
                self.best_weights = model
        return False

class FraudDetector:
    def __init__(self, n_epochs=50, early_stopping_patience=10, regularization_strength=0.1,
                 sampling_strategy='auto', use_smote=True):
        self.model = None
        self.scaler = RobustScaler()  # Better for outliers in fraud detection
        self.feature_names = []
        self.model_metrics = {}
        self.training_history = {
            'train_auc': [],
            'val_auc': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'epoch_times': []
        }
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.regularization_strength = regularization_strength
        self.sampling_strategy = sampling_strategy
        self.use_smote = use_smote
        
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced feature engineering for fraud detection
        """
        df = data.copy()
        
        # Time-based features
        df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
        df['Day_of_week'] = (df['Time'] // (24 * 3600)) % 7
        df['Is_weekend'] = (df['Day_of_week'] >= 5).astype(int)
        df['Is_night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
        
        # Amount-based features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Amount categories
        df['Amount_category'] = pd.cut(df['Amount'], 
                                     bins=[0, 10, 50, 200, 500, 2000, float('inf')],
                                     labels=[0, 1, 2, 3, 4, 5]).astype(int)
        
        # High-risk amount ranges (common fraud patterns)
        df['High_risk_amount'] = ((df['Amount'] < 5) | 
                                 ((df['Amount'] >= 100) & (df['Amount'] <= 300)) |
                                 (df['Amount'] >= 2000)).astype(int)
        
        # V features interactions (PCA components)
        # Create interaction features between important V features
        v_features = [col for col in df.columns if col.startswith('V')]
        
        # V feature aggregations
        df['V_sum'] = df[v_features].sum(axis=1)
        df['V_mean'] = df[v_features].mean(axis=1)
        df['V_std'] = df[v_features].std(axis=1)
        df['V_max'] = df[v_features].max(axis=1)
        df['V_min'] = df[v_features].min(axis=1)
        
        # V feature outlier detection
        for feature in ['V1', 'V2', 'V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']:
            if feature in df.columns:
                q75, q25 = np.percentile(df[feature], [75, 25])
                iqr = q75 - q25
                outlier_threshold = 1.5 * iqr
                df[f'{feature}_outlier'] = ((df[feature] < (q25 - outlier_threshold)) | 
                                          (df[feature] > (q75 + outlier_threshold))).astype(int)
        
        # Key V feature interactions (based on fraud detection research)
        df['V1_V2_interaction'] = df['V1'] * df['V2']
        df['V3_V4_interaction'] = df['V3'] * df['V4']
        df['V7_V20_interaction'] = df['V7'] * df['V20'] if 'V20' in df.columns else 0
        df['V12_V16_interaction'] = df['V12'] * df['V16'] if 'V16' in df.columns else 0
        
        # Frequency-based features (transaction patterns)
        df['Amount_frequency'] = df.groupby('Amount')['Amount'].transform('count')
        
        # Risk score based on multiple factors
        df['Risk_score'] = (df['High_risk_amount'] * 0.3 + 
                           df['Is_night'] * 0.2 + 
                           df['Is_weekend'] * 0.1 +
                           (df['Amount_log'] / df['Amount_log'].max()) * 0.4)
        
        return df
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Advanced preprocessing for fraud detection with class imbalance handling
        """
        print("🔧 Engineering fraud detection features...")
        df = self.engineer_features(data)
        
        # Remove original Time column (we've extracted features from it)
        feature_columns = [col for col in df.columns if col not in ['Time', 'Class']]
        
        # Prepare features and target
        X = df[feature_columns].copy()
        y = df['Class'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        # Scale features (RobustScaler is better for fraud detection with outliers)
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        self.feature_names = feature_columns
        
        # Print class distribution
        class_counts = y.value_counts()
        fraud_rate = class_counts[1] / len(y) * 100
        print(f"✅ Features engineered: {len(feature_columns)} features, {len(X_scaled)} samples")
        print(f"📊 Class distribution: Normal={class_counts[0]}, Fraud={class_counts[1]} ({fraud_rate:.2f}%)")
        
        return X_scaled, y
    
    def create_ensemble_model(self) -> Pipeline:
        """
        Create advanced ensemble classifier optimized for fraud detection
        """
        # XGBoost with class weight balancing
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_epochs,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=self.regularization_strength,
            reg_lambda=self.regularization_strength,
            scale_pos_weight=99,  # Adjust for class imbalance (99:1 ratio typical)
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )
        
        # Random Forest with balanced class weights
        rf_model = RandomForestClassifier(
            n_estimators=self.n_epochs,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',  # Handle imbalanced classes
            random_state=42,
            n_jobs=-1
        )
        
        # Voting ensemble
        ensemble = VotingClassifier([
            ('xgb', xgb_model),
            ('rf', rf_model)
        ], voting='soft')
        
        # Create pipeline with optional SMOTE
        if self.use_smote:
            pipeline = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=self.sampling_strategy, random_state=42)),
                ('classifier', ensemble)
            ])
        else:
            pipeline = Pipeline([
                ('classifier', ensemble)
            ])
        
        return pipeline
    
    def train_model_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train ensemble model with stratified cross-validation optimized for fraud detection
        """
        print(f"🚀 Starting fraud detection training with {self.n_epochs} epochs...")
        print(f"🔄 Using {'SMOTE oversampling' if self.use_smote else 'class weights'} for imbalance handling")
        
        # Stratified K-Fold cross-validation (maintains class proportions)
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {'auc': [], 'precision': [], 'recall': [], 'f1': []}
        
        print(f"📊 Performing {cv_folds}-fold stratified cross-validation...")
        
        # Cross-validation with progress bar
        cv_progress = tqdm(skf.split(X, y), total=cv_folds, desc="CV Folds")
        
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
            
            # Evaluate fold with fraud-specific metrics
            val_pred = fold_model.predict(X_val_fold)
            val_pred_proba = fold_model.predict_proba(X_val_fold)[:, 1]
            
            fold_auc = roc_auc_score(y_val_fold, val_pred_proba)
            fold_precision = precision_score(y_val_fold, val_pred, zero_division=0)
            fold_recall = recall_score(y_val_fold, val_pred)
            fold_f1 = f1_score(y_val_fold, val_pred)
            
            cv_scores['auc'].append(fold_auc)
            cv_scores['precision'].append(fold_precision)
            cv_scores['recall'].append(fold_recall)
            cv_scores['f1'].append(fold_f1)
            
            cv_progress.set_postfix({
                'Fold': fold + 1,
                'AUC': f'{fold_auc:.4f}',
                'Precision': f'{fold_precision:.4f}',
                'Recall': f'{fold_recall:.4f}'
            })
        
        cv_progress.close()
        
        # Final train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Hyperparameter grid optimized for fraud detection
        param_grid = {
            'classifier__xgb__n_estimators': [50, 100, 150],
            'classifier__xgb__max_depth': [4, 6, 8],
            'classifier__xgb__learning_rate': [0.05, 0.1, 0.15],
            'classifier__xgb__reg_alpha': [0.01, 0.1, 0.2],
            'classifier__xgb__scale_pos_weight': [50, 99, 150],
            'classifier__rf__n_estimators': [50, 100],
            'classifier__rf__max_depth': [8, 10, 12]
        }
        
        # Adjust parameter names if using SMOTE pipeline
        if self.use_smote:
            param_grid = {f"classifier__{k.split('__', 1)[1]}": v 
                         for k, v in param_grid.items()}
        
        base_model = self.create_ensemble_model()
        
        print("🔍 Performing hyperparameter optimization for fraud detection...")
        
        # Grid search with AUC scoring (better for imbalanced classes)
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        # Training with progress tracking
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.model = grid_search.best_estimator_
        
        # Comprehensive evaluation on test set
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_train = self.model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate comprehensive fraud detection metrics
        self.model_metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train),
            'test_recall': recall_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train),
            'test_f1': f1_score(y_test, y_pred_test),
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'test_auc': roc_auc_score(y_test, y_pred_proba_test),
            'train_ap': average_precision_score(y_train, y_pred_proba_train),
            'test_ap': average_precision_score(y_test, y_pred_proba_test),
            'cv_scores': cv_scores,
            'cv_mean_auc': np.mean(cv_scores['auc']),
            'cv_std_auc': np.std(cv_scores['auc']),
            'cv_mean_precision': np.mean(cv_scores['precision']),
            'cv_mean_recall': np.mean(cv_scores['recall']),
            'best_params': grid_search.best_params_,
            'grid_search_score': grid_search.best_score_,
            'training_time': training_time,
            'n_epochs': self.n_epochs,
            'regularization_strength': self.regularization_strength,
            'use_smote': self.use_smote,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
            'classification_report': classification_report(y_test, y_pred_test, 
                                                         target_names=['Normal', 'Fraud'], 
                                                         output_dict=True)
        }
        
        # Store training history
        self.training_history['train_auc'].append(self.model_metrics['train_auc'])
        self.training_history['val_auc'].append(self.model_metrics['test_auc'])
        self.training_history['train_precision'].append(self.model_metrics['train_precision'])
        self.training_history['val_precision'].append(self.model_metrics['test_precision'])
        self.training_history['train_recall'].append(self.model_metrics['train_recall'])
        self.training_history['val_recall'].append(self.model_metrics['test_recall'])
        self.training_history['epoch_times'].append(training_time)
        
        logger.info(f"✅ Fraud detection model trained successfully!")
        logger.info(f"Test AUC: {self.model_metrics['test_auc']:.4f}")
        logger.info(f"Test Precision: {self.model_metrics['test_precision']:.4f}")
        logger.info(f"Test Recall: {self.model_metrics['test_recall']:.4f}")
        
        return self.model
    
    def plot_training_history(self, save_path: str = 'fraud_training_history.png'):
        """Plot comprehensive training history and evaluation metrics"""
        if not self.training_history['train_auc']:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC plot
        axes[0, 0].plot(self.training_history['train_auc'], label='Train AUC', marker='o')
        axes[0, 0].plot(self.training_history['val_auc'], label='Validation AUC', marker='s')
        axes[0, 0].set_title('Fraud Detection Model AUC')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('AUC Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision plot
        axes[0, 1].plot(self.training_history['train_precision'], label='Train Precision', marker='o')
        axes[0, 1].plot(self.training_history['val_precision'], label='Validation Precision', marker='s')
        axes[0, 1].set_title('Fraud Detection Model Precision')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Recall plot
        axes[1, 0].plot(self.training_history['train_recall'], label='Train Recall', marker='o')
        axes[1, 0].plot(self.training_history['val_recall'], label='Validation Recall', marker='s')
        axes[1, 0].set_title('Fraud Detection Model Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Confusion Matrix
        cm = np.array(self.model_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'],
                   ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Fraud detection training history saved to {save_path}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make fraud predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Use the same preprocessing pipeline
        features, _ = self.preprocess_data(data)
        
        return self.model.predict(features)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get fraud prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Use the same preprocessing pipeline
        features, _ = self.preprocess_data(data)
        
        return self.model.predict_proba(features)
    
    def get_fraud_risk_score(self, data: pd.DataFrame) -> np.ndarray:
        """Get fraud risk scores (probability of fraud)"""
        return self.predict_proba(data)[:, 1]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        if self.model is None:
            return {}
        
        try:
            # Handle different pipeline structures
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
            else:
                classifier = self.model
            
            # Extract feature importance from ensemble components
            if hasattr(classifier, 'estimators'):
                xgb_classifier = classifier.estimators[0]
                rf_classifier = classifier.estimators[1]
                
                # Average importance from both models
                xgb_importance = xgb_classifier.feature_importances_
                rf_importance = rf_classifier.feature_importances_
                
                avg_importance = (xgb_importance + rf_importance) / 2
                
                importance_dict = dict(zip(self.feature_names, avg_importance))
                
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str = 'fraud_detector_model.pkl') -> bool:
        """Save the trained model and preprocessors"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': self.model_metrics,
                'training_history': self.training_history,
                'hyperparameters': {
                    'n_epochs': self.n_epochs,
                    'early_stopping_patience': self.early_stopping_patience,
                    'regularization_strength': self.regularization_strength,
                    'sampling_strategy': self.sampling_strategy,
                    'use_smote': self.use_smote
                },
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str = 'fraud_detector_model.pkl') -> bool:
        """Load a trained model and preprocessors"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler', RobustScaler())
            self.feature_names = model_data.get('feature_names', [])
            self.model_metrics = model_data.get('metrics', {})
            self.training_history = model_data.get('training_history', {})
            
            # Load hyperparameters if available
            hyperparams = model_data.get('hyperparameters', {})
            self.n_epochs = hyperparams.get('n_epochs', 50)
            self.early_stopping_patience = hyperparams.get('early_stopping_patience', 10)
            self.regularization_strength = hyperparams.get('regularization_strength', 0.1)
            self.sampling_strategy = hyperparams.get('sampling_strategy', 'auto')
            self.use_smote = hyperparams.get('use_smote', True)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def train_fraud_detection_model(data_path: str = '../creditcard.csv',
                               n_epochs: int = 50,
                               early_stopping_patience: int = 10,
                               regularization_strength: float = 0.1,
                               use_smote: bool = True,
                               sampling_strategy: str = 'auto',
                               sample_size: int = 100000,  # Limit for memory management
                               save_model: bool = True,
                               model_path: str = 'fraud_detector.pkl') -> FraudDetector:
    """
    Train and return a fraud detection model using credit card data with advanced features
    """
    # Load credit card dataset
    try:
        data = pd.read_csv(data_path)
        print(f"🏦 Loaded credit card dataset with {len(data)} records and {len(data.columns)} features")
        
        # Check class distribution
        class_counts = data['Class'].value_counts()
        fraud_rate = class_counts[1] / len(data) * 100
        print(f"📊 Original class distribution: Normal={class_counts[0]}, Fraud={class_counts[1]} ({fraud_rate:.3f}%)")
        
        # Sample data for memory management if specified
        if sample_size and sample_size < len(data):
            # Stratified sampling to maintain class balance
            normal_sample = data[data['Class'] == 0].sample(n=int(sample_size * (1 - fraud_rate/100)), random_state=42)
            fraud_sample = data[data['Class'] == 1].sample(n=int(sample_size * (fraud_rate/100)), random_state=42)
            data = pd.concat([normal_sample, fraud_sample], ignore_index=True).sample(frac=1, random_state=42)
            print(f"🎯 Using stratified sample of {len(data)} records for training")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    detector = FraudDetector(
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        regularization_strength=regularization_strength,
        sampling_strategy=sampling_strategy,
        use_smote=use_smote
    )
    
    # Preprocess data
    print("🔄 Preprocessing fraud detection data...")
    X, y = detector.preprocess_data(data)
    
    # Train model with advanced features
    detector.train_model_with_cv(X, y)
    
    # Save model if requested
    if save_model:
        detector.save_model(model_path)
    
    # Print comprehensive results
    metrics = detector.model_metrics
    print(f"\n{'='*50}")
    print(f"🏦 FRAUD DETECTION MODEL RESULTS")
    print(f"{'='*50}")
    print(f"📊 Performance Metrics:")
    print(f"   • Test AUC: {metrics['test_auc']:.4f}")
    print(f"   • Test Precision: {metrics['test_precision']:.4f}")
    print(f"   • Test Recall: {metrics['test_recall']:.4f}")
    print(f"   • Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"   • Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   • Average Precision: {metrics['test_ap']:.4f}")
    print(f"   • Training Time: {metrics['training_time']:.2f}s")
    print(f"\n🔀 Cross-Validation:")
    print(f"   • CV Mean AUC: {metrics['cv_mean_auc']:.4f} ± {metrics['cv_std_auc']:.4f}")
    print(f"   • CV Mean Precision: {metrics['cv_mean_precision']:.4f}")
    print(f"   • CV Mean Recall: {metrics['cv_mean_recall']:.4f}")
    print(f"\n⚙️  Training Configuration:")
    print(f"   • Epochs: {metrics['n_epochs']}")
    print(f"   • Regularization: {metrics['regularization_strength']}")
    print(f"   • SMOTE Sampling: {metrics['use_smote']}")
    print(f"   • Early Stopping: {early_stopping_patience} patience")
    
    # Detailed classification report
    report = metrics['classification_report']
    print(f"\n🏷️  Detailed Performance:")
    for class_name, scores in report.items():
        if isinstance(scores, dict) and 'f1-score' in scores:
            print(f"   • {class_name}: Precision={scores['precision']:.3f}, "
                  f"Recall={scores['recall']:.3f}, F1={scores['f1-score']:.3f}, "
                  f"Support={scores['support']}")
    
    # Feature importance
    importance = detector.get_feature_importance()
    if importance:
        print(f"\n🔍 Top 15 Important Features:")
        for i, (feature, score) in enumerate(list(importance.items())[:15]):
            print(f"   {i+1:2d}. {feature:<25}: {score:.4f}")
    
    # Plot training history and metrics
    detector.plot_training_history()
    
    return detector

# Backward compatibility functions
def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Legacy function for backward compatibility"""
    detector = FraudDetector()
    return detector.preprocess_data(data)

def train_model(X: pd.DataFrame, y: pd.Series) -> Any:
    """Legacy function for backward compatibility"""
    detector = FraudDetector()
    return detector.train_model_with_cv(X, y)

if __name__ == "__main__":
    print("🏦 Training fraud detection model on credit card dataset...")
    detector = train_fraud_detection_model(
        sample_size=100000,  # Use 100k samples for demo
        n_epochs=50,
        early_stopping_patience=10,
        regularization_strength=0.1,
        use_smote=True
    )
    
    if detector:
        print(f"\n✅ Fraud detection model training completed successfully!")
        print(f"📁 Model saved as: fraud_detector.pkl")
        print(f"📈 Training history plot saved as: fraud_training_history.png")
        print(f"🛡️  Model ready for real-time fraud detection!") 