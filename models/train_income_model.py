import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any, Optional, List
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

class IncomeClassifier:
    def __init__(self, n_epochs=50, early_stopping_patience=10, regularization_strength=0.1):
        self.model = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.model_metrics = {}
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
        Advanced preprocessing for Adult dataset income classification
        """
        df = data.copy()
        
        # Handle missing values (marked as '?' in adult dataset)
        df = df.replace('?', np.nan)
        
        # Define feature columns for Adult dataset
        numerical_features = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 
                            'capital-loss', 'hours-per-week']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                              'relationship', 'race', 'gender', 'native-country']
        
        # Handle missing values
        for col in numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
        
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Advanced feature engineering
        df['age_squared'] = df['age'] ** 2
        df['capital_total'] = df['capital-gain'] - df['capital-loss']
        df['education_hours_interaction'] = df['educational-num'] * df['hours-per-week']
        df['age_education_interaction'] = df['age'] * df['educational-num']
        df['hours_per_week_normalized'] = df['hours-per-week'] / 168  # Weekly hours normalization
        
        # Age groups with more granular categories
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 25, 35, 45, 55, 65, 100], 
                               labels=['young', 'early_career', 'mid_career', 'senior', 'pre_retirement', 'retirement'])
        
        # Working hours categories
        df['hours_category'] = pd.cut(df['hours-per-week'],
                                    bins=[0, 20, 40, 50, 60, 100],
                                    labels=['part_time', 'full_time', 'standard_overtime', 'high_overtime', 'workaholic'])
        
        # Education level grouping
        education_mapping = {
            'Preschool': 'low', '1st-4th': 'low', '5th-6th': 'low', '7th-8th': 'low',
            '9th': 'medium', '10th': 'medium', '11th': 'medium', '12th': 'medium',
            'HS-grad': 'medium', 'Some-college': 'medium',
            'Assoc-voc': 'high', 'Assoc-acdm': 'high', 'Bachelors': 'high',
            'Masters': 'very_high', 'Prof-school': 'very_high', 'Doctorate': 'very_high'
        }
        df['education_level'] = df['education'].map(education_mapping).fillna('medium')
        
        # Update feature lists with engineered features
        numerical_features.extend(['age_squared', 'capital_total', 'education_hours_interaction',
                                 'age_education_interaction', 'hours_per_week_normalized'])
        categorical_features.extend(['age_group', 'hours_category', 'education_level'])
        
        # Create preprocessor with regularization-friendly scaling
        if self.preprocessor is None:
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
        
        # Prepare features and target
        X = df[numerical_features + categorical_features]
        y = df['income']
        
        # Transform features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Create feature names for interpretability
        num_feature_names = numerical_features
        cat_feature_names = []
        
        if hasattr(self.preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
        else:
            cat_feature_names = [f'cat_{i}' for i in range(X_processed.shape[1] - len(numerical_features))]
        
        self.feature_names = num_feature_names + cat_feature_names
        
        # Convert to DataFrame for consistency
        X_processed = pd.DataFrame(X_processed, columns=self.feature_names)
        
        return X_processed, y
    
    def create_ensemble_model(self) -> Pipeline:
        """
        Create advanced ensemble classifier with regularization
        """
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
            eval_metric='logloss'
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=self.n_epochs,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
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
        print(f"🚀 Starting training with {self.n_epochs} epochs and early stopping...")
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Stratified K-Fold cross-validation
        cv_folds = 5
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        print(f"📊 Performing {cv_folds}-fold cross-validation...")
        
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
            fold_score = f1_score(y_val_fold, val_pred, average='weighted')
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
        
        # Hyperparameter grid with regularization focus
        param_grid = {
            'classifier__xgb__n_estimators': [50, 100, 150],
            'classifier__xgb__max_depth': [4, 6, 8],
            'classifier__xgb__learning_rate': [0.05, 0.1, 0.15],
            'classifier__xgb__reg_alpha': [0.01, 0.1, 0.2],
            'classifier__xgb__reg_lambda': [0.01, 0.1, 0.2],
            'classifier__rf__n_estimators': [50, 100],
            'classifier__rf__max_depth': [8, 10, 12],
            'classifier__rf__min_samples_split': [5, 10, 15]
        }
        
        base_model = self.create_ensemble_model()
        
        print("🔍 Performing hyperparameter optimization...")
        
        # Grid search with progress tracking
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=3, scoring='f1_weighted',  # Reduced CV for speed
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
            'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
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
        
        logger.info(f"✅ Model trained successfully!")
        logger.info(f"Test Accuracy: {self.model_metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1: {self.model_metrics['test_f1']:.4f}")
        logger.info(f"CV Mean: {self.model_metrics['cv_mean']:.4f} ± {self.model_metrics['cv_std']:.4f}")
        
        return self.model
    
    def plot_training_history(self, save_path: str = 'income_training_history.png'):
        """Plot training history"""
        if not self.training_history['train_loss']:
            print("No training history to plot")
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss', marker='s')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['train_accuracy'], label='Train Accuracy', marker='o')
        ax2.plot(self.training_history['val_accuracy'], label='Validation Accuracy', marker='s')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Training history saved to {save_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make income predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        predictions_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        if self.model is None:
            return {}
        
        # Extract feature importance from ensemble components
        xgb_classifier = self.model.named_steps['classifier'].estimators[0]
        rf_classifier = self.model.named_steps['classifier'].estimators[1]
        
        # Average importance from both models
        xgb_importance = xgb_classifier.feature_importances_
        rf_importance = rf_classifier.feature_importances_
        
        avg_importance = (xgb_importance + rf_importance) / 2
        
        importance_dict = dict(zip(self.feature_names, avg_importance))
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath: str = 'income_classifier_model.pkl') -> bool:
        """Save the trained model and preprocessors"""
        try:
            model_data = {
                'model': self.model,
                'preprocessor': self.preprocessor,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
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
    
    def load_model(self, filepath: str = 'income_classifier_model.pkl') -> bool:
        """Load a trained model and preprocessors"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.preprocessor = model_data.get('preprocessor', None)
            self.label_encoder = model_data.get('label_encoder', LabelEncoder())
            self.feature_names = model_data.get('feature_names', [])
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

def train_income_classification_model(data_path: str = '../adult.csv',
                                    n_epochs: int = 50,
                                    early_stopping_patience: int = 10,
                                    regularization_strength: float = 0.1,
                                    sample_size: int = 5000,  # Limit sample size for faster training
                                    save_model: bool = True,
                                    model_path: str = 'income_classifier.pkl') -> IncomeClassifier:
    """
    Train and return an income classification model using Adult dataset with advanced features
    """
    # Load Adult dataset
    try:
        data = pd.read_csv(data_path)
        print(f"📁 Loaded dataset with {len(data)} records and {len(data.columns)} features")
        
        # Sample data for faster training if specified
        if sample_size and sample_size < len(data):
            data = data.sample(n=sample_size, random_state=42)
            print(f"🎯 Using sample of {len(data)} records for training")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    classifier = IncomeClassifier(
        n_epochs=n_epochs,
        early_stopping_patience=early_stopping_patience,
        regularization_strength=regularization_strength
    )
    
    # Preprocess data
    print("🔄 Preprocessing data...")
    X, y = classifier.preprocess_data(data)
    
    # Train model with advanced features
    classifier.train_model_with_cv(X, y)
    
    # Save model if requested
    if save_model:
        classifier.save_model(model_path)
    
    # Print comprehensive results
    metrics = classifier.model_metrics
    print(f"\n{'='*50}")
    print(f"🎯 INCOME CLASSIFICATION MODEL RESULTS")
    print(f"{'='*50}")
    print(f"📊 Performance Metrics:")
    print(f"   • Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"   • Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"   • Test Loss: {metrics['test_loss']:.4f}")
    print(f"   • Training Time: {metrics['training_time']:.2f}s")
    print(f"\n🔀 Cross-Validation:")
    print(f"   • CV Mean F1: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print(f"   • All CV Scores: {[f'{s:.3f}' for s in metrics['cv_scores']]}")
    print(f"\n⚙️  Training Configuration:")
    print(f"   • Epochs: {metrics['n_epochs']}")
    print(f"   • Regularization: {metrics['regularization_strength']}")
    print(f"   • Early Stopping: {early_stopping_patience} patience")
    
    # Class performance
    report = metrics['classification_report']
    print(f"\n📈 Income Level Performance:")
    for income_level, scores in report.items():
        if isinstance(scores, dict) and 'f1-score' in scores:
            print(f"   • {income_level}: Precision={scores['precision']:.3f}, "
                  f"Recall={scores['recall']:.3f}, F1={scores['f1-score']:.3f}, "
                  f"Support={scores['support']}")
    
    # Feature importance
    importance = classifier.get_feature_importance()
    print(f"\n🔍 Top 10 Important Features:")
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"   {i+1:2d}. {feature:<25}: {score:.4f}")
    
    # Plot training history
    classifier.plot_training_history()
    
    return classifier

# Backward compatibility functions
def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Legacy function for backward compatibility"""
    classifier = IncomeClassifier()
    return classifier.preprocess_data(data)

def train_model(X: pd.DataFrame, y: pd.Series) -> Any:
    """Legacy function for backward compatibility"""
    classifier = IncomeClassifier()
    return classifier.train_model_with_cv(X, y)

if __name__ == "__main__":
    print("🚀 Training income classification model on Adult dataset...")
    classifier = train_income_classification_model(
        sample_size=10000,  # Use 10k samples for demo
        n_epochs=50,
        early_stopping_patience=10,
        regularization_strength=0.1
    )
    
    if classifier:
        # Test prediction on limited samples
        print("\n🧪 Testing predictions on sample data...")
        test_data = pd.read_csv('../adult.csv').head(5)  # Only 5 samples for testing
        X_test, y_true = classifier.preprocess_data(test_data)
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        
        print(f"\n{'='*60}")
        print(f"🎯 SAMPLE PREDICTIONS")
        print(f"{'='*60}")
        for i in range(len(predictions)):
            max_prob = np.max(probabilities[i])
            print(f"👤 Person {i+1}:")
            print(f"   Age: {test_data.iloc[i]['age']}, Education: {test_data.iloc[i]['education']}")
            print(f"   Occupation: {test_data.iloc[i]['occupation']}")
            print(f"   Actual: {y_true.iloc[i]} | Predicted: {predictions[i]} (confidence: {max_prob:.3f})")
            print(f"   {'-'*50}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model saved as: income_classifier.pkl")
        print(f"📈 Training history plot saved as: income_training_history.png") 