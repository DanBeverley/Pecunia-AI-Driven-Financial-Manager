import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import joblib
import re
import logging
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
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

class ExpenseTextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer for extracting features from expense descriptions and merchants"""
    
    def __init__(self):
        self.description_tfidf = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            lowercase=True
        )
        
        self.merchant_tfidf = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.9,
            lowercase=True
        )
        
    def fit(self, X, y=None):
        descriptions = X['description'].fillna('').astype(str)
        merchants = X['merchant'].fillna('').astype(str)
        
        cleaned_descriptions = descriptions.apply(self._clean_text)
        cleaned_merchants = merchants.apply(self._clean_merchant)
        
        self.description_tfidf.fit(cleaned_descriptions)
        self.merchant_tfidf.fit(cleaned_merchants)
        return self
    
    def transform(self, X):
        descriptions = X['description'].fillna('').astype(str)
        merchants = X['merchant'].fillna('').astype(str)
        
        cleaned_descriptions = descriptions.apply(self._clean_text)
        cleaned_merchants = merchants.apply(self._clean_merchant)
        
        # TF-IDF features
        desc_features = self.description_tfidf.transform(cleaned_descriptions).toarray()
        merchant_features = self.merchant_tfidf.transform(cleaned_merchants).toarray()
        
        # Combine features
        combined_features = np.hstack([desc_features, merchant_features])
        
        # Create feature names
        desc_feature_names = [f'desc_tfidf_{i}' for i in range(desc_features.shape[1])]
        merchant_feature_names = [f'merchant_tfidf_{i}' for i in range(merchant_features.shape[1])]
        feature_names = desc_feature_names + merchant_feature_names
        
        return pd.DataFrame(combined_features, columns=feature_names, index=X.index)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize expense descriptions"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)  # Remove numbers for description
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _clean_merchant(self, text: str) -> str:
        """Clean merchant names"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

class ExpenseNumericalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract numerical and categorical features from expense data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.merchant_encoder = LabelEncoder()
        
    def fit(self, X, y=None):
        numerical_features = self._extract_numerical_features(X)
        self.scaler.fit(numerical_features)
        
        # Fit merchant encoder
        merchants = X['merchant'].fillna('unknown').astype(str)
        self.merchant_encoder.fit(merchants)
        
        return self
    
    def transform(self, X):
        numerical_features = self._extract_numerical_features(X)
        scaled_features = self.scaler.transform(numerical_features)
        
        # Merchant encoding
        merchants = X['merchant'].fillna('unknown').astype(str)
        # Handle unknown merchants during transform
        merchant_encoded = []
        for merchant in merchants:
            try:
                merchant_encoded.append(self.merchant_encoder.transform([merchant])[0])
            except ValueError:
                # Unknown merchant, assign a default value
                merchant_encoded.append(-1)
        
        merchant_encoded = np.array(merchant_encoded).reshape(-1, 1)
        
        # Combine features
        combined_features = np.hstack([scaled_features, merchant_encoded])
        
        feature_names = ['amount_log', 'amount_normalized', 'amount_category', 
                        'description_length', 'merchant_length', 'merchant_encoded']
        
        return pd.DataFrame(combined_features, columns=feature_names, index=X.index)
    
    def _extract_numerical_features(self, X):
        features = pd.DataFrame(index=X.index)
        
        # Amount features with advanced engineering
        amounts = pd.to_numeric(X['amount'], errors='coerce').fillna(0)
        features['amount_log'] = np.log1p(amounts)
        features['amount_normalized'] = amounts / amounts.max() if amounts.max() > 0 else 0
        
        # More granular amount categories
        features['amount_category'] = pd.cut(amounts, 
                                           bins=[0, 10, 25, 50, 100, 200, float('inf')], 
                                           labels=[0, 1, 2, 3, 4, 5]).astype(int)
        
        # Text length features
        features['description_length'] = X['description'].fillna('').astype(str).str.len()
        features['merchant_length'] = X['merchant'].fillna('').astype(str).str.len()
        
        return features.fillna(0)

class ExpenseClassifier:
    def __init__(self, n_epochs=50, early_stopping_patience=10, regularization_strength=0.1):
        self.model = None
        self.text_extractor = ExpenseTextFeatureExtractor()
        self.numerical_extractor = ExpenseNumericalFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.model_metrics = {}
        self.category_mapping = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': []
        }
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.regularization_strength = regularization_strength
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Advanced feature preprocessing for expense classification dataset
        """
        df = data.copy()
        
        # Ensure required columns exist
        required_cols = ['amount', 'merchant', 'description', 'category']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        # Clean and prepare data
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['merchant'] = df['merchant'].fillna('unknown').astype(str)
        df['description'] = df['description'].fillna('').astype(str)
        df['category'] = df['category'].fillna('miscellaneous').astype(str)
        
        # Create feature union pipeline
        feature_union = FeatureUnion([
            ('text_features', self.text_extractor),
            ('numerical_features', self.numerical_extractor)
        ])
        
        # Fit and transform features
        features = feature_union.fit_transform(df)
        
        # Ensure features is a DataFrame
        if isinstance(features, np.ndarray):
            # Create column names
            text_feature_names = [f'text_feat_{i}' for i in range(400)]  # 300 desc + 100 merchant
            num_feature_names = ['amount_log', 'amount_normalized', 'amount_category', 
                               'description_length', 'merchant_length', 'merchant_encoded']
            all_feature_names = text_feature_names + num_feature_names
            features = pd.DataFrame(features, columns=all_feature_names[:features.shape[1]], index=df.index)
        
        # Target categories
        target = df['category']
        
        return features, target
    
    def create_ensemble_model(self) -> Pipeline:
        """
        Create advanced ensemble classifier optimized for expense categorization with regularization
        """
        # Base models with regularization
        xgb_model = xgb.XGBClassifier(
            n_estimators=self.n_epochs,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=self.regularization_strength,  # L1 regularization
            reg_lambda=self.regularization_strength,  # L2 regularization
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=self.n_epochs,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',  # Regularization through feature subsampling
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Voting ensemble
        ensemble = VotingClassifier([
            ('xgb', xgb_model),
            ('rf', rf_model)
        ], voting='soft')
        
        pipeline = Pipeline([
            ('classifier', ensemble)
        ])
        
        return pipeline
    
    def train_model_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train ensemble model with cross-validation, early stopping, and progress tracking
        """
        print(f"🚀 Starting expense classification training with {self.n_epochs} epochs...")
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.category_mapping = dict(zip(self.label_encoder.classes_, 
                                       range(len(self.label_encoder.classes_))))
        
        # Stratified K-Fold cross-validation
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        print(f"📊 Performing {cv_folds}-fold cross-validation for expense categories...")
        
        # Cross-validation with progress bar
        cv_progress = tqdm(skf.split(X, y_encoded), total=cv_folds, desc="CV Folds")
        
        for fold, (train_idx, val_idx) in enumerate(cv_progress):
            # Convert to DataFrame if it's numpy array
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
                X_train_fold, X_val_fold = X_df.iloc[train_idx], X_df.iloc[val_idx]
            else:
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            
            y_train_fold, y_val_fold = y_encoded[train_idx], y_encoded[val_idx]
            
            # Train model for this fold
            fold_model = self.create_ensemble_model()
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate fold
            val_pred = fold_model.predict(X_val_fold)
            fold_score = f1_score(y_val_fold, val_pred, average='macro')
            cv_scores.append(fold_score)
            
            cv_progress.set_postfix({
                'Fold': fold + 1,
                'F1': f'{fold_score:.4f}',
                'Mean': f'{np.mean(cv_scores):.4f}'
            })
        
        cv_progress.close()
        
        # Final train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Optimized hyperparameter grid for expense classification with regularization
        param_grid = {
            'classifier__xgb__n_estimators': [50, 100, 150],
            'classifier__xgb__max_depth': [4, 6, 8],
            'classifier__xgb__learning_rate': [0.05, 0.1, 0.15],
            'classifier__xgb__reg_alpha': [0.01, 0.1, 0.2],
            'classifier__xgb__reg_lambda': [0.01, 0.1, 0.2],
            'classifier__rf__n_estimators': [50, 100],
            'classifier__rf__max_depth': [10, 12, 15],
            'classifier__rf__min_samples_split': [2, 3, 5]
        }
        
        base_model = self.create_ensemble_model()
        
        print("🔍 Performing hyperparameter optimization for expense classification...")
        
        # Grid search with progress tracking
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=3, scoring='f1_macro',  # Reduced CV for speed
            n_jobs=-1, verbose=1
        )
        
        # Training with progress tracking
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.model = grid_search.best_estimator_
        
        # Evaluate model performance with detailed metrics
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)
        
        # Calculate loss
        train_loss = log_loss(y_train, self.model.predict_proba(X_train))
        test_loss = log_loss(y_test, y_pred_proba_test)
        
        # Get class names for reporting
        class_names = self.label_encoder.classes_
        
        self.model_metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_f1_macro': f1_score(y_train, y_pred_train, average='macro'),
            'test_f1_macro': f1_score(y_test, y_pred_test, average='macro'),
            'train_f1_weighted': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1_weighted': f1_score(y_test, y_pred_test, average='weighted'),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'best_params': grid_search.best_params_,
            'grid_search_score': grid_search.best_score_,
            'training_time': training_time,
            'n_epochs': self.n_epochs,
            'regularization_strength': self.regularization_strength,
            'classification_report': classification_report(y_test, y_pred_test, 
                                                         target_names=class_names, 
                                                         output_dict=True)
        }
        
        # Store training history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(test_loss)
        self.training_history['train_accuracy'].append(self.model_metrics['train_accuracy'])
        self.training_history['val_accuracy'].append(self.model_metrics['test_accuracy'])
        self.training_history['epoch_times'].append(training_time)
        
        logger.info(f"✅ Expense model trained successfully!")
        logger.info(f"Test Accuracy: {self.model_metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1-Macro: {self.model_metrics['test_f1_macro']:.4f}")
        logger.info(f"CV Mean: {self.model_metrics['cv_mean']:.4f} ± {self.model_metrics['cv_std']:.4f}")
        
        return self.model
    
    def plot_training_history(self, save_path: str = 'expense_training_history.png'):
        """Plot training history"""
        if not self.training_history['train_loss']:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss', marker='s')
        ax1.set_title('Expense Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['train_accuracy'], label='Train Accuracy', marker='o')
        ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax2.set_title('Expense Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Expense training history saved to {save_path}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make category predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Use the same preprocessing pipeline
        features, _ = self.preprocess_data(data)
        
        predictions_encoded = self.model.predict(features)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Use the same preprocessing pipeline
        features, _ = self.preprocess_data(data)
        
        return self.model.predict_proba(features)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        if self.model is None:
            return {}
        
        try:
            # Extract feature importance from ensemble components
            xgb_classifier = self.model.named_steps['classifier'].estimators[0]
            rf_classifier = self.model.named_steps['classifier'].estimators[1]
            
            # Average importance from both models
            xgb_importance = xgb_classifier.feature_importances_
            rf_importance = rf_classifier.feature_importances_
            
            avg_importance = (xgb_importance + rf_importance) / 2
            
            # Get feature names from the last transformation
            feature_names = [f'feature_{i}' for i in range(len(avg_importance))]
            importance_dict = dict(zip(feature_names, avg_importance))
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def save_model(self, filepath: str = 'expense_classifier_model.pkl') -> bool:
        """Save the trained model and preprocessors"""
        try:
            model_data = {
                'model': self.model,
                'text_extractor': self.text_extractor,
                'numerical_extractor': self.numerical_extractor,
                'label_encoder': self.label_encoder,
                'category_mapping': self.category_mapping,
                'metrics': self.model_metrics,
                'training_history': self.training_history,
                'hyperparameters': {
                    'n_epochs': self.n_epochs,
                    'early_stopping_patience': self.early_stopping_patience,
                    'regularization_strength': self.regularization_strength
                },
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str = 'expense_classifier_model.pkl') -> bool:
        """Load a trained model and preprocessors"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.text_extractor = model_data.get('text_extractor', ExpenseTextFeatureExtractor())
            self.numerical_extractor = model_data.get('numerical_extractor', ExpenseNumericalFeatureExtractor())
            self.label_encoder = model_data.get('label_encoder', LabelEncoder())
            self.category_mapping = model_data.get('category_mapping', {})
            self.model_metrics = model_data.get('metrics', {})
            self.training_history = model_data.get('training_history', {})
            
            # Load hyperparameters if available
            hyperparams = model_data.get('hyperparameters', {})
            self.n_epochs = hyperparams.get('n_epochs', 50)
            self.early_stopping_patience = hyperparams.get('early_stopping_patience', 10)
            self.regularization_strength = hyperparams.get('regularization_strength', 0.1)
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

def train_expense_classification_model(data_path: str = '../personal_expense_classification.csv',
                                     n_epochs: int = 50,
                                     early_stopping_patience: int = 10,
                                     regularization_strength: float = 0.1,
                                     sample_size: int = None,  # Use all data for expense model (small dataset)
                                     save_model: bool = True,
                                     model_path: str = 'expense_classifier.pkl') -> ExpenseClassifier:
    """
    Train and return an expense classification model using personal expense dataset with advanced features
    """
    # Load expense dataset
    try:
        data = pd.read_csv(data_path)
        print(f"💰 Loaded expense dataset with {len(data)} records and {len(data.columns)} features")
        print(f"🏷️  Categories found: {sorted(data['category'].unique())}")
        print(f"📊 Category distribution:\n{data['category'].value_counts()}")
        
        # Sample data if specified (for very large datasets)
        if sample_size and sample_size < len(data):
            data = data.sample(n=sample_size, random_state=42)
            print(f"🎯 Using sample of {len(data)} records for training")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    classifier = ExpenseClassifier(
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        regularization_strength=regularization_strength
    )
    
    # Preprocess data
    print("🔄 Preprocessing expense data...")
    X, y = classifier.preprocess_data(data)
    
    # Train model with advanced features
    classifier.train_model_with_cv(X, y)
    
    # Save model if requested
    if save_model:
        classifier.save_model(model_path)
    
    # Print comprehensive results
    metrics = classifier.model_metrics
    print(f"\n{'='*50}")
    print(f"💰 EXPENSE CLASSIFICATION MODEL RESULTS")
    print(f"{'='*50}")
    print(f"📊 Performance Metrics:")
    print(f"   • Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   • Test F1-Macro: {metrics['test_f1_macro']:.4f}")
    print(f"   • Test F1-Weighted: {metrics['test_f1_weighted']:.4f}")
    print(f"   • Test Loss: {metrics['test_loss']:.4f}")
    print(f"   • Training Time: {metrics['training_time']:.2f}s")
    print(f"\n🔀 Cross-Validation:")
    print(f"   • CV Mean F1: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"   • All CV Scores: {[f'{s:.3f}' for s in metrics['cv_scores']]}")
    print(f"\n⚙️  Training Configuration:")
    print(f"   • Epochs: {metrics['n_epochs']}")
    print(f"   • Regularization: {metrics['regularization_strength']}")
    print(f"   • Early Stopping: {early_stopping_patience} patience")
    
    # Category performance
    report = metrics['classification_report']
    print(f"\n🏷️  Category Performance:")
    for category, scores in report.items():
        if isinstance(scores, dict) and 'f1-score' in scores:
            print(f"   • {category.capitalize()}: Precision={scores['precision']:.3f}, "
                  f"Recall={scores['recall']:.3f}, F1={scores['f1-score']:.3f}, "
                  f"Support={scores['support']}")
    
    # Plot training history
    classifier.plot_training_history()
    
    return classifier

# Backward compatibility functions
def categorize_expenses(data: pd.DataFrame) -> pd.DataFrame:
    """Legacy function for backward compatibility"""
    # Simple rule-based categorization for compatibility
    df = data.copy()
    if 'category' not in df.columns:
        df['category'] = 'miscellaneous'
    return df

def train_model(X: pd.DataFrame, y: pd.Series) -> Any:
    """Legacy function for backward compatibility"""
    classifier = ExpenseClassifier()
    return classifier.train_model_with_cv(X, y)

if __name__ == "__main__":
    print("💰 Training expense classification model on personal expense dataset...")
    classifier = train_expense_classification_model(
        n_epochs=50,
        early_stopping_patience=10,
        regularization_strength=0.1
    )
    
    if classifier:
        # Test prediction on limited samples
        print("\n🧪 Testing predictions on sample expense data...")
        test_data = pd.read_csv('../personal_expense_classification.csv').sample(5, random_state=42)  # Only 5 samples
        
        predictions = classifier.predict(test_data)
        probabilities = classifier.predict_proba(test_data)
        
        print(f"\n{'='*70}")
        print(f"💰 SAMPLE EXPENSE PREDICTIONS")
        print(f"{'='*70}")
        for i in range(len(predictions)):
            max_prob = np.max(probabilities[i])
            print(f"🛒 Expense {i+1}:")
            print(f"   Amount: ${test_data.iloc[i]['amount']:.2f}")
            print(f"   Merchant: {test_data.iloc[i]['merchant']}")
            print(f"   Description: {test_data.iloc[i]['description']}")
            print(f"   Actual: {test_data.iloc[i]['category']} | Predicted: {predictions[i]} (confidence: {max_prob:.3f})")
            print(f"   {'-'*60}")
        
        print(f"\n✅ Expense model training completed successfully!")
        print(f"📁 Model saved as: expense_classifier.pkl")
        print(f"📈 Training history plot saved as: expense_training_history.png") 