import numpy as np
import pandas as pd
import yaml
import joblib
import os
import time
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Advanced model quantization system for ML models"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.compression_report = {}
        self.original_models = {}
        self.quantized_models = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load quantization configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Config loading failed: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for quantization"""
        return {
            'quantization': {
                'global': {'enable_quantization': True, 'default_strategy': 'balanced'},
                'models': {
                    'income_classifier': {'strategy': 'aggressive', 'expected_compression': 0.25},
                    'expense_classifier': {'strategy': 'conservative', 'expected_compression': 0.4},
                    'investment_predictor': {'strategy': 'balanced', 'expected_compression': 0.35},
                    'fraud_detector': {'strategy': 'precision_focused', 'expected_compression': 0.3}
                }
            }
        }
    
    def quantize_all_models(self, models_dir: str = ".", output_dir: str = "quantized") -> Dict[str, Any]:
        """Quantize all available models"""
        Path(output_dir).mkdir(exist_ok=True)
        results = {}
        
        # Find all pickle files
        model_files = list(Path(models_dir).glob("*.pkl"))
        
        print(f"Found {len(model_files)} model files for quantization")
        
        for model_file in tqdm(model_files, desc="Quantizing models"):
            model_name = self._extract_model_name(model_file.name)
            if model_name in self.config['quantization']['models']:
                try:
                    original_model = joblib.load(model_file)
                    quantized_model, compression_stats = self._quantize_model(
                        original_model, model_name
                    )
                    
                    # Save quantized model
                    output_path = Path(output_dir) / f"{model_file.stem}_quantized.pkl"
                    joblib.dump(quantized_model, output_path)
                    
                    results[model_name] = {
                        'original_path': str(model_file),
                        'quantized_path': str(output_path),
                        'compression_stats': compression_stats
                    }
                    
                    print(f"{model_name}: {compression_stats['size_reduction']:.1%} size reduction")
                    
                except Exception as e:
                    logger.error(f"Failed to quantize {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
        
        # Generate comprehensive report
        self._generate_quantization_report(results, output_dir)
        return results
    
    def _extract_model_name(self, filename: str) -> str:
        """Extract model name from filename"""
        mapping = {
            'income_classifier': 'income_classifier',
            'expense_classifier': 'expense_classifier', 
            'investment_predictor': 'investment_predictor',
            'fraud_detector': 'fraud_detector'
        }
        
        for key, value in mapping.items():
            if key in filename.lower():
                return value
        return 'unknown'
    
    def _quantize_model(self, model: Any, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Apply quantization to specific model type"""
        config = self.config['quantization']['models'][model_name]
        
        # Get original model size
        original_size = self._get_model_size(model)
        
        # Apply quantization based on model type
        if 'classifier' in model_name:
            quantized_model = self._quantize_classifier(model, config)
        elif 'predictor' in model_name:
            quantized_model = self._quantize_regressor(model, config)
        else:
            quantized_model = self._generic_quantization(model, config)
        
        # Calculate compression statistics
        quantized_size = self._get_model_size(quantized_model)
        compression_stats = {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'size_reduction': 1 - (quantized_size / original_size),
            'compression_ratio': original_size / quantized_size
        }
        
        return quantized_model, compression_stats
    
    def _quantize_classifier(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize classification models"""
        if hasattr(model, 'named_steps'):  # Pipeline
            return self._quantize_pipeline_classifier(model, config)
        elif hasattr(model, 'estimators_'):  # Ensemble
            return self._quantize_ensemble_classifier(model, config)
        else:
            return self._quantize_single_classifier(model, config)
    
    def _quantize_regressor(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize regression models"""
        if hasattr(model, 'named_steps'):  # Pipeline
            return self._quantize_pipeline_regressor(model, config)
        elif hasattr(model, 'estimators_'):  # Ensemble
            return self._quantize_ensemble_regressor(model, config)
        else:
            return self._quantize_single_regressor(model, config)
    
    def _quantize_ensemble_classifier(self, ensemble: Any, config: Dict[str, Any]) -> Any:
        """Quantize ensemble classifiers"""
        if isinstance(ensemble, VotingClassifier):
            # Reduce number of estimators based on config
            reduction_ratio = config.get('parameters', {}).get('estimator_reduction', 0.5)
            n_keep = max(1, int(len(ensemble.estimators_) * (1 - reduction_ratio)))
            
            # Keep best performing estimators
            if hasattr(ensemble, 'estimators_'):
                estimators_to_keep = ensemble.estimators_[:n_keep]
                ensemble.estimators_ = estimators_to_keep
                ensemble.named_estimators_ = dict(list(ensemble.named_estimators_.items())[:n_keep])
        
        # Quantize individual estimators
        if hasattr(ensemble, 'estimators_'):
            for i, estimator in enumerate(ensemble.estimators_):
                ensemble.estimators_[i] = self._quantize_tree_model(estimator, config)
        
        return ensemble
    
    def _quantize_ensemble_regressor(self, ensemble: Any, config: Dict[str, Any]) -> Any:
        """Quantize ensemble regressors"""
        if isinstance(ensemble, VotingRegressor):
            # Similar logic as classifier
            reduction_ratio = config.get('parameters', {}).get('estimator_reduction', 0.5)
            n_keep = max(1, int(len(ensemble.estimators_) * (1 - reduction_ratio)))
            
            if hasattr(ensemble, 'estimators_'):
                estimators_to_keep = ensemble.estimators_[:n_keep]
                ensemble.estimators_ = estimators_to_keep
                ensemble.named_estimators_ = dict(list(ensemble.named_estimators_.items())[:n_keep])
        
        # Quantize individual estimators
        if hasattr(ensemble, 'estimators_'):
            for i, estimator in enumerate(ensemble.estimators_):
                ensemble.estimators_[i] = self._quantize_tree_model(estimator, config)
        
        return ensemble
    
    def _quantize_tree_model(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize tree-based models (XGBoost, RandomForest)"""
        if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            return self._quantize_xgboost(model, config)
        elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
            return self._quantize_random_forest(model, config)
        return model
    
    def _quantize_xgboost(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize XGBoost models"""
        # Reduce number of estimators
        if hasattr(model, 'n_estimators'):
            reduction_ratio = config.get('parameters', {}).get('estimator_reduction', 0.3)
            new_n_estimators = max(10, int(model.n_estimators * (1 - reduction_ratio)))
            model.n_estimators = new_n_estimators
        
        # Reduce max_depth
        if hasattr(model, 'max_depth'):
            depth_reduction = config.get('parameters', {}).get('max_depth_reduction', 0.2)
            if model.max_depth:
                new_depth = max(3, int(model.max_depth * (1 - depth_reduction)))
                model.max_depth = new_depth
        
        return model
    
    def _quantize_random_forest(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize Random Forest models"""
        # Reduce number of estimators
        if hasattr(model, 'estimators_'):
            reduction_ratio = config.get('parameters', {}).get('estimator_reduction', 0.3)
            n_keep = max(10, int(len(model.estimators_) * (1 - reduction_ratio)))
            model.estimators_ = model.estimators_[:n_keep]
            model.n_estimators = len(model.estimators_)
        
        return model
    
    def _quantize_pipeline_classifier(self, pipeline: Any, config: Dict[str, Any]) -> Any:
        """Quantize pipeline classifiers"""
        # Quantize each step
        for name, step in pipeline.named_steps.items():
            if hasattr(step, 'estimators_'):  # Ensemble
                pipeline.named_steps[name] = self._quantize_ensemble_classifier(step, config)
            elif isinstance(step, TfidfVectorizer):
                pipeline.named_steps[name] = self._quantize_tfidf(step, config)
        
        return pipeline
    
    def _quantize_pipeline_regressor(self, pipeline: Any, config: Dict[str, Any]) -> Any:
        """Quantize pipeline regressors"""
        # Similar to classifier pipeline
        for name, step in pipeline.named_steps.items():
            if hasattr(step, 'estimators_'):
                pipeline.named_steps[name] = self._quantize_ensemble_regressor(step, config)
            elif isinstance(step, TfidfVectorizer):
                pipeline.named_steps[name] = self._quantize_tfidf(step, config)
        
        return pipeline
    
    def _quantize_tfidf(self, tfidf: TfidfVectorizer, config: Dict[str, Any]) -> TfidfVectorizer:
        """Quantize TF-IDF vectorizers"""
        params = config.get('parameters', {})
        
        # Reduce max_features
        if hasattr(tfidf, 'max_features') and tfidf.max_features:
            reduction_ratio = params.get('max_tfidf_features', 150)
            if isinstance(reduction_ratio, int):
                tfidf.max_features = min(tfidf.max_features, reduction_ratio)
        
        # Increase min_df to reduce vocabulary
        if hasattr(tfidf, 'min_df'):
            min_df_multiplier = params.get('min_df_increase', 2)
            tfidf.min_df = max(tfidf.min_df * min_df_multiplier, 2)
        
        return tfidf
    
    def _quantize_single_classifier(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize single classifier models"""
        return self._quantize_tree_model(model, config)
    
    def _quantize_single_regressor(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize single regressor models"""
        return self._quantize_tree_model(model, config)
    
    def _generic_quantization(self, model: Any, config: Dict[str, Any]) -> Any:
        """Generic quantization for unknown model types"""
        return model  # Return as-is for unknown types
    
    def _get_model_size(self, model: Any) -> int:
        """Get model size in bytes"""
        import pickle
        return len(pickle.dumps(model))
    
    def benchmark_models(self, models_dict: Dict[str, Any], X_test: pd.DataFrame, 
                        y_test: pd.Series, model_type: str = 'classifier') -> Dict[str, Any]:
        """Benchmark original vs quantized models"""
        results = {}
        
        for name, model_info in models_dict.items():
            if 'error' in model_info:
                continue
                
            try:
                original_model = joblib.load(model_info['original_path'])
                quantized_model = joblib.load(model_info['quantized_path'])
                
                # Benchmark performance
                original_perf = self._evaluate_model(original_model, X_test, y_test, model_type)
                quantized_perf = self._evaluate_model(quantized_model, X_test, y_test, model_type)
                
                # Benchmark speed
                original_time = self._benchmark_inference_speed(original_model, X_test)
                quantized_time = self._benchmark_inference_speed(quantized_model, X_test)
                
                results[name] = {
                    'performance': {
                        'original': original_perf,
                        'quantized': quantized_perf,
                        'performance_retention': quantized_perf / original_perf if original_perf > 0 else 0
                    },
                    'speed': {
                        'original_ms': original_time * 1000,
                        'quantized_ms': quantized_time * 1000,
                        'speedup': original_time / quantized_time if quantized_time > 0 else 0
                    },
                    'compression': model_info['compression_stats']
                }
                
            except Exception as e:
                logger.error(f"Benchmarking failed for {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                       model_type: str) -> float:
        """Evaluate model performance"""
        try:
            if model_type == 'classifier':
                if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                    # Binary classification - use AUC
                    y_proba = model.predict_proba(X_test)[:, 1]
                    return roc_auc_score(y_test, y_proba)
                else:
                    # Multi-class - use accuracy
                    y_pred = model.predict(X_test)
                    return accuracy_score(y_test, y_pred)
            else:  # regressor
                y_pred = model.predict(X_test)
                return r2_score(y_test, y_pred)
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return 0.0
    
    def _benchmark_inference_speed(self, model: Any, X_test: pd.DataFrame, n_runs: int = 100) -> float:
        """Benchmark model inference speed"""
        times = []
        
        # Warm up
        for _ in range(5):
            _ = model.predict(X_test[:10])
        
        # Actual benchmarking
        for _ in range(n_runs):
            start_time = time.time()
            _ = model.predict(X_test[:100])  # Test on 100 samples
            times.append(time.time() - start_time)
        
        return np.mean(times)
    
    def _generate_quantization_report(self, results: Dict[str, Any], output_dir: str):
        """Generate comprehensive quantization report"""
        report_path = Path(output_dir) / "quantization_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Model Quantization Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Compression Summary\n\n")
            f.write("| Model | Original Size (MB) | Quantized Size (MB) | Size Reduction | Compression Ratio |\n")
            f.write("|-------|-------------------|-------------------|----------------|------------------|\n")
            
            for name, result in results.items():
                if 'compression_stats' in result:
                    stats = result['compression_stats']
                    f.write(f"| {name} | {stats['original_size_mb']:.2f} | "
                           f"{stats['quantized_size_mb']:.2f} | "
                           f"{stats['size_reduction']:.1%} | "
                           f"{stats['compression_ratio']:.1f}x |\n")
            
            f.write("\n## Quantization Details\n\n")
            for name, result in results.items():
                if 'error' not in result:
                    f.write(f"### {name.replace('_', ' ').title()}\n")
                    f.write(f"- **Strategy**: {self.config['quantization']['models'][name]['strategy']}\n")
                    f.write(f"- **Original Path**: `{result['original_path']}`\n")
                    f.write(f"- **Quantized Path**: `{result['quantized_path']}`\n")
                    f.write(f"- **Size Reduction**: {result['compression_stats']['size_reduction']:.1%}\n\n")
        
        print(f"Quantization report saved to: {report_path}")

class QuantizationPipeline:
    """End-to-end quantization pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.quantizer = ModelQuantizer(config_path)
        
    def run_full_quantization(self, models_dir: str = ".", output_dir: str = "quantized") -> Dict[str, Any]:
        """Run complete quantization pipeline"""
        print("Starting Model Quantization Pipeline...")
        
        # Step 1: Quantize all models
        quantization_results = self.quantizer.quantize_all_models(models_dir, output_dir)
        
        # Step 2: Generate visualizations
        self._create_compression_visualizations(quantization_results, output_dir)
        
        print("Quantization pipeline completed successfully!")
        return quantization_results
    
    def _create_compression_visualizations(self, results: Dict[str, Any], output_dir: str):
        """Create visualization plots for compression results"""
        models = []
        original_sizes = []
        quantized_sizes = []
        reductions = []
        
        for name, result in results.items():
            if 'compression_stats' in result:
                models.append(name.replace('_', '\n').title())
                stats = result['compression_stats']
                original_sizes.append(stats['original_size_mb'])
                quantized_sizes.append(stats['quantized_size_mb'])
                reductions.append(stats['size_reduction'] * 100)
        
        if not models:
            return
        
        # Create subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🚀 Model Quantization Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Size comparison
        x_pos = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x_pos - width/2, original_sizes, width, label='Original', alpha=0.8, color='#ff7f0e')
        ax1.bar(x_pos + width/2, quantized_sizes, width, label='Quantized', alpha=0.8, color='#2ca02c')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Size (MB)')
        ax1.set_title('Model Size Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Size reduction percentage
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax2.bar(models, reductions, color=colors, alpha=0.8)
        ax2.set_ylabel('Size Reduction (%)')
        ax2.set_title('Compression Efficiency')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, reduction in zip(bars, reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Compression ratio
        compression_ratios = [result['compression_stats']['compression_ratio'] 
                            for result in results.values() if 'compression_stats' in result]
        
        ax3.pie(compression_ratios, labels=models, autopct='%1.1fx', startangle=90,
               colors=colors, explode=[0.05] * len(models))
        ax3.set_title('Compression Ratios')
        
        # Plot 4: Summary metrics
        ax4.axis('off')
        total_original = sum(original_sizes)
        total_quantized = sum(quantized_sizes)
        overall_reduction = (1 - total_quantized / total_original) * 100
        
        summary_text = f"""
        📊 QUANTIZATION SUMMARY
        
        Total Original Size: {total_original:.1f} MB
        Total Quantized Size: {total_quantized:.1f} MB
        Overall Size Reduction: {overall_reduction:.1f}%
        
        🎯 Models Quantized: {len(models)}
        💾 Space Saved: {total_original - total_quantized:.1f} MB
        ⚡ Average Compression: {np.mean(compression_ratios):.1f}x
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'quantization_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {Path(output_dir) / 'quantization_visualization.png'}")

from datetime import datetime 