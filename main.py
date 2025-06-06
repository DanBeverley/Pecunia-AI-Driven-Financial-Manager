#!/usr/bin/env python3
"""
Pecunia AI - Enterprise Financial Management System
Main entry point with model training, detection, and Streamlit app launch
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import warnings
from sklearn.exceptions import ConvergenceWarning

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging with Windows encoding fix
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pecunia.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our enterprise utils
try:
    from utils import (
        initialize_all_systems, SecurityPolicy, AuthMethod, UserRole,
        clean_data, detect_anomalies, BlockchainNetwork
    )
except ImportError:
    logger.warning("Some utils modules not available - continuing with limited functionality")

def check_model_exists(model_path: str) -> bool:
    """Check if a trained model exists"""
    return Path(model_path).exists()

def detect_existing_models() -> Dict[str, bool]:
    """Detect which models are already trained"""
    model_paths = {
        'expense': 'models/expense_classifier.pkl', 
        'income': 'models/income_classifier.pkl',
        'investment': 'models/investment_predictor.pkl'
    }
    
    existing_models = {}
    for model_name, model_path in model_paths.items():
        exists = check_model_exists(model_path)
        existing_models[model_name] = exists
        if exists:
            logger.info(f"✅ Found existing {model_name} model: {model_path}")
        else:
            logger.info(f"❌ Missing {model_name} model: {model_path}")
    
    return existing_models

def train_missing_models(missing_models: list, args: argparse.Namespace):
    """Train missing models using the actual training functions"""
    if not missing_models:
        logger.info("🎯 All models are already trained!")
        return
    
    logger.info(f"🧠 Training {len(missing_models)} missing models...")
    
    # Define training functions that actually work
    training_functions = {}
    
    if 'expense' in missing_models:
        try:
            from models.train_expense_model import train_expense_classification_model
            training_functions['expense'] = lambda: train_expense_classification_model(
                data_path='personal_expense_classification.csv',
                n_epochs=args.epochs,
                sample_size=None,  # Use all data for expense (small dataset)
                save_model=True,
                model_path='models/expense_classifier.pkl'
            )
        except ImportError as e:
            logger.error(f"Failed to import expense training: {e}")
    
    if 'income' in missing_models:
        try:
            from models.train_income_model import train_income_classification_model
            training_functions['income'] = lambda: train_income_classification_model(
                data_path='adult.csv',
                n_epochs=args.epochs,
                sample_size=args.sample_size,  # Use sample_size for large dataset
                save_model=True,
                model_path='models/income_classifier.pkl'
            )
        except ImportError as e:
            logger.error(f"Failed to import income training: {e}")
    
    if 'investment' in missing_models:
        try:
            from models.train_investment_model import train_investment_prediction_model
            training_functions['investment'] = lambda: train_investment_prediction_model(
                data_path='all_stocks_5yr.csv',
                n_epochs=args.epochs,
                sample_size=args.sample_size,  # Use sample_size for large dataset
                save_model=True,
                model_path='models/investment_predictor.pkl'
            )
        except ImportError as e:
            logger.error(f"Failed to import investment training: {e}")
    
    # Train models
    successful_models = 0
    total_training_time = 0
    
    for i, model_name in enumerate(missing_models, 1):
        if model_name in training_functions:
            logger.info(f"🚀 Training {model_name} model ({i}/{len(missing_models)})...")
            logger.info(f"   📊 Using {args.sample_size} samples, {args.epochs} epochs")
            
            try:
                start_time = time.time()
                model = training_functions[model_name]()
                train_time = time.time() - start_time
                total_training_time += train_time
                
                if model:
                    logger.info(f"✅ {model_name.title()} model trained in {train_time:.2f}s")
                    successful_models += 1
                    
                    # Quick verification
                    model_path = f"models/{model_name}_classifier.pkl"
                    if model_name == 'investment':
                        model_path = f"models/{model_name}_predictor.pkl"
                    
                    if os.path.exists(model_path):
                        logger.info(f"   💾 Model saved: {model_path}")
                    else:
                        logger.warning(f"   ⚠️ Model file not found: {model_path}")
                else:
                    logger.error(f"❌ Failed to train {model_name} model - returned None")
                    
            except Exception as e:
                logger.error(f"❌ Error training {model_name}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            logger.warning(f"⚠️ No training function for {model_name}")
        
        # Brief pause between models for visibility
        if i < len(missing_models):
            time.sleep(1)
    
    # Summary
    logger.info(f"🏁 Training completed: {successful_models}/{len(missing_models)} successful")
    logger.info(f"⏱️  Total time: {total_training_time:.2f}s")

def setup_production_environment():
    """Setup production environment with all features enabled"""
    logger.info("🔧 Setting up production environment...")
    
    # Create necessary directories
    required_dirs = ['models', 'utils', 'data', 'logs', 'app']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            logger.info(f"📁 Created directory: {dir_name}")
    
    # Ensure log file exists
    log_file = Path('logs/pecunia.log')
    if not log_file.exists():
        log_file.touch()
    
    # Set production API keys and settings
    production_settings = {
        'NEWSAPI_KEY': 'your_newsapi_key_here',  # Replace with actual key
        'ALPHA_VANTAGE_KEY': 'IIFJKUOT0B0M7OV6',  # Already provided
        'OPENAI_API_KEY': 'your_openai_key_here',  # Replace with actual key
        'POLYGON_API_KEY': 'your_polygon_key_here',  # Replace with actual key
        'ENVIRONMENT': 'production',
        'DEBUG': 'False',
        'ENABLE_GPU': 'False',  # Set to True if GPU available
        'MODEL_CACHE': 'True',
        'AUTO_REFRESH': 'True'
    }
    
    # Write environment settings
    env_file = Path('.env')
    if not env_file.exists():
        with open(env_file, 'w') as f:
            for key, value in production_settings.items():
                f.write(f"{key}={value}\n")
        logger.info("📝 Created .env file with production settings")
    
    logger.info("✅ Production environment setup complete!")

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing production-ready dependencies...")
    
    try:
        # Install app dependencies
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'app/requirements.txt'
        ], check=True, capture_output=True)
        logger.info("✅ App dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        
        # Install production-ready packages for Bayesian optimization
        critical_packages = [
            'streamlit>=1.28.0',
            'pandas>=1.5.0', 
            'numpy>=1.24.0',
            'plotly>=5.15.0',
            'requests>=2.31.0',
            'scikit-learn>=1.3.0',
            'scikit-optimize>=0.10.0',  # Bayesian optimization
            'optuna>=4.0.0',            # Advanced hyperparameter tuning
            'xgboost>=2.0.0',           # Modern XGBoost
            'psutil>=5.9.0',            # System monitoring
            'GPUtil>=1.4.0'             # GPU monitoring
        ]
        
        for package in critical_packages:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], check=True, capture_output=True)
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {package}")

def launch_streamlit_app():
    """Launch the Streamlit application with proper browser support"""
    logger.info("🚀 Launching Pecunia AI Streamlit Application...")
    
    app_path = Path('app/pecunia_app.py')
    if not app_path.exists():
        logger.error(f"❌ Streamlit app not found: {app_path}")
        return False
    
    try:
        # Launch Streamlit with browser-compatible settings
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            str(app_path),
            '--server.port=8501',
            '--server.address=127.0.0.1',  # Fixed: Use 127.0.0.1 instead of 0.0.0.0
            '--server.headless=false',
            '--browser.gatherUsageStats=false',
            '--browser.serverAddress=localhost',  # Ensure browser connects properly
            '--theme.primaryColor=#00ff88',
            '--theme.backgroundColor=#1a1a1a',    # Dark theme
            '--theme.secondaryBackgroundColor=#2d2d2d'
        ]
        
        logger.info("🌐 Starting Streamlit server...")
        logger.info("📱 Access the app at: http://localhost:8501")
        logger.info("🎨 Theme: Dark mode enabled")
        
        # Auto-open browser after short delay
        import webbrowser
        import threading
        
        def open_browser():
            time.sleep(3)  # Wait for server to start
            try:
                webbrowser.open("http://localhost:8501")
                logger.info("🚀 Browser opened automatically")
            except Exception as e:
                logger.warning(f"⚠️ Could not open browser: {e}")
        
        # Start browser opening in background
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        process = subprocess.run(cmd)
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"❌ Failed to launch Streamlit app: {e}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Pecunia AI - Complete Financial Management System with Bayesian Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Production Features:
  • Bayesian hyperparameter optimization (96 iterations)
  • GPU-accelerated training with CUDA support
  • Comprehensive system monitoring & logging
  • Intelligent model detection & skipping
  • Production-grade error handling & fallbacks

Examples:
  # Run complete system (recommended)
  python main.py
  
  # Run with custom training settings
  python main.py --sample-size 50000 --epochs 30
  
  # Skip training and just run app
  python main.py --skip-training
  
  # Force retrain all models with Bayesian optimization
  python main.py --force-retrain
  
  # Production deployment with all features
  python main.py --production --install-deps
        """
    )
    
    # Training options
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1000,  # Debug: Reduced from 10000 to 1000
        help='Sample size for training (default: 1000 for debugging)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,  # Debug: Reduced from 20 to 3
        help='Number of training epochs (default: 3 for debugging)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training and go directly to app'
    )
    
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain all models even if they exist'
    )
    
    # Deployment options
    parser.add_argument(
        '--production',
        action='store_true',
        help='Setup production environment'
    )
    
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install dependencies before running'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port for Streamlit server (default: 8501)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host for Streamlit server (default: 0.0.0.0)'
    )
    
    # System options  
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with minimal samples'
    )
    
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    return parser.parse_args()

def print_banner():
    """Print the Pecunia AI banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        PECUNIA AI                            ║
    ║              AI-Driven Financial Management                  ║
    ║                                                              ║
    ║  📰 Smart Newsfeed       🎓 Financial Education            ║
    ║  💬 Community Forum      📊 AI Analytics                  ║  
    ║  🤖 ML Models           🚀 Real-time Insights              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point for complete Pecunia AI system"""
    args = parse_arguments()
    
    print_banner()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔧 Verbose mode enabled")
    
    if args.debug:
        # Override for debug mode
        args.sample_size = min(args.sample_size, 500)  # Max 500 samples in debug
        args.epochs = min(args.epochs, 2)  # Max 2 epochs in debug
        logger.info("🐛 DEBUG MODE: Using minimal samples and epochs")
        logger.info(f"   • Sample size: {args.sample_size}")
        logger.info(f"   • Epochs: {args.epochs}")
        
        # Suppress non-critical warnings ONLY in debug mode
        warnings.filterwarnings("ignore", category=UserWarning, message="Features.*are constant.")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        logger.info("   • Suppressing non-critical UserWarning, RuntimeWarning, and ConvergenceWarning.")
    
    logger.info("🚀 Starting Pecunia AI Complete System...")
    
    try:
        # Step 1: Setup production environment if requested
        if args.production:
            logger.info("⚙️  Setting up production environment...")
            setup_production_environment()
        
        # Step 2: Install dependencies if requested
        if args.install_deps:
            logger.info("📦 Installing dependencies...")
            install_dependencies()
        
        # Step 3: Model training and detection (MUST complete before app launch)
        if not args.skip_training:
            logger.info("🧠 Checking model status...")
            existing_models = detect_existing_models()
            
            # Print detailed model status
            logger.info("📊 Model Status:")
            for model_name, exists in existing_models.items():
                status = "✅ Found" if exists else "❌ Missing"
                logger.info(f"  • {model_name}: {status}")
            
            if args.force_retrain:
                logger.info("🔄 Force retraining all models...")
                missing_models = ['expense', 'income', 'investment']
            else:
                missing_models = [name for name, exists in existing_models.items() if not exists]
            
            if missing_models:
                logger.info(f"🎯 Starting training for: {', '.join(missing_models)}")
                logger.info("⏳ Training in progress... Please wait until completion!")
                
                # BLOCKING TRAINING - Must complete before proceeding
                train_missing_models(missing_models, args)
                
                # Verify training completion
                logger.info("🔍 Verifying training completion...")
                final_models_check = detect_existing_models()
                trained_count = sum(final_models_check.values())
                total_count = len(final_models_check)
                
                logger.info(f"📊 Training Results: {trained_count}/{total_count} models available")
                
                if trained_count == total_count:
                    logger.info("✅ All models trained successfully!")
                else:
                    logger.warning(f"⚠️  Only {trained_count}/{total_count} models completed")
            else:
                logger.info("✅ All models are available!")
        else:
            logger.info("⏭️ Skipping model training as requested")
        
        # Step 4: Launch Streamlit application (ONLY after training completes)
        logger.info("🌟 All preparation complete - launching application...")
        logger.info("🚀 Starting Streamlit server in 3 seconds...")
        
        # Brief countdown for visibility
        for i in range(3, 0, -1):
            logger.info(f"⏳ {i}...")
            time.sleep(1)
        
        if not args.no_browser:
            success = launch_streamlit_app()
            
            if success:
                logger.info("🎉 Pecunia AI launched successfully!")
            else:
                logger.error("❌ Failed to launch Pecunia AI")
        else:
            logger.info("🚫 Browser launch disabled by user")
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Operation cancelled by user")
    except Exception as e:
        logger.error(f"\n❌ Error in main execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    logger.info("\n🚀 Pecunia AI - Ready for Production!")
    return 0

if __name__ == "__main__":
    # Run the main function
    exit(main()) 