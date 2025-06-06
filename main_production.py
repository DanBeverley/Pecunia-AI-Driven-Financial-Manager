#!/usr/bin/env python3
"""
Pecunia AI - Production Deployment Script
Complete system with model detection, training, and Streamlit app launch
"""

import argparse
import asyncio
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Ensure logs directory exists
logs_dir = Path('logs')
if not logs_dir.exists():
    logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pecunia.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_model_exists(model_path: str) -> bool:
    """Check if a trained model exists"""
    return Path(model_path).exists()

def detect_existing_models() -> Dict[str, bool]:
    """Detect which models are already trained"""
    model_paths = {
        'fraud': 'models/fraud_detector.pkl',
        'expense': 'models/expense_classifier.pkl', 
        'investment': 'models/investment_model.pkl',
        'income': 'models/income_predictor.pkl'
    }
    
    existing_models = {}
    for model_name, model_path in model_paths.items():
        exists = check_model_exists(model_path)
        existing_models[model_name] = exists
        if exists:
            logger.info(f"[OK] Found existing {model_name} model: {model_path}")
        else:
            logger.info(f"[MISSING] {model_name} model: {model_path}")
    
    return existing_models

def train_missing_models(missing_models: list, sample_size: int = 10000, epochs: int = 20):
    """Train only the missing models"""
    if not missing_models:
        logger.info("[INFO] All models are already trained!")
        return
    
    logger.info(f"[TRAINING] {len(missing_models)} missing models...")
    
    # Import model training functions
    training_functions = {}
    
    try:
        from models.fraud_detection import train_fraud_detection_model
        training_functions['fraud'] = lambda: train_fraud_detection_model(
            data_path='creditcard.csv',
            sample_size=sample_size,
            n_epochs=epochs,
            save_model=True,
            model_path='models/fraud_detector.pkl'
        )
    except ImportError:
        logger.warning("Fraud detection training not available")
    
    try:
        from models.train_expense_model import train_expense_classification_model
        training_functions['expense'] = lambda: train_expense_classification_model(
            data_path='personal_expense_classification.csv',
            sample_size=sample_size,
            n_epochs=epochs,
            save_model=True,
            model_path='models/expense_classifier.pkl'
        )
    except ImportError:
        logger.warning("Expense classification training not available")
    
    try:
        from models.train_investment_model import train_investment_prediction_model
        training_functions['investment'] = lambda: train_investment_prediction_model(
            data_path='all_stocks_5yr.csv',
            sample_size=sample_size,
            n_epochs=epochs,
            save_model=True,
            model_path='models/investment_model.pkl'
        )
    except ImportError:
        logger.warning("Investment model training not available")
    
    try:
        from models.train_income_model import train_income_classification_model
        training_functions['income'] = lambda: train_income_classification_model(
            data_path='adult.csv',
            sample_size=sample_size,
            n_epochs=epochs,
            save_model=True,
            model_path='models/income_predictor.pkl'
        )
    except ImportError:
        logger.warning("Income prediction training not available")
    
    # Train missing models
    for model_name in missing_models:
        if model_name in training_functions:
            logger.info(f"[TRAINING] {model_name} model...")
            try:
                start_time = time.time()
                model = training_functions[model_name]()
                train_time = time.time() - start_time
                
                if model:
                    logger.info(f"[SUCCESS] {model_name.title()} model trained successfully in {train_time:.2f}s")
                else:
                    logger.error(f"[ERROR] Failed to train {model_name} model")
            except Exception as e:
                logger.error(f"[ERROR] Error training {model_name} model: {e}")
        else:
            logger.warning(f"[WARNING] Training function not available for {model_name}")

def setup_production_environment():
    """Setup production environment with all features enabled"""
    logger.info("[SETUP] Setting up production environment...")
    
    # Create necessary directories
    required_dirs = ['models', 'utils', 'data', 'logs', 'app']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            logger.info(f"[CREATE] Created directory: {dir_name}")
    
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
        logger.info("[CONFIG] Created .env file with production settings")
    
    logger.info("[SUCCESS] Production environment setup complete!")

def install_dependencies():
    """Install required dependencies"""
    logger.info("[INSTALL] Installing dependencies...")
    
    try:
        # Install app dependencies
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'app/requirements.txt'
        ], check=True, capture_output=True)
        logger.info("[SUCCESS] App dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to install dependencies: {e}")
        # Try installing individual critical packages
        critical_packages = [
            'streamlit>=1.28.0',
            'pandas>=1.5.0', 
            'numpy>=1.24.0',
            'plotly>=5.15.0',
            'requests>=2.31.0',
            'scikit-learn>=1.3.0'
        ]
        
        for package in critical_packages:
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], check=True, capture_output=True)
                logger.info(f"[OK] Installed {package}")
            except subprocess.CalledProcessError:
                logger.warning(f"[WARNING] Failed to install {package}")

def launch_streamlit_app():
    """Launch the Streamlit application"""
    logger.info("[LAUNCH] Launching Pecunia AI Streamlit Application...")
    
    app_path = Path('app/pecunia_app.py')
    if not app_path.exists():
        logger.error(f"[ERROR] Streamlit app not found: {app_path}")
        return False
    
    try:
        # Launch Streamlit with localhost for browser compatibility
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            str(app_path),
            '--server.port=8501',
            '--server.address=localhost',  # Changed from 0.0.0.0 to localhost
            '--server.headless=false',
            '--browser.gatherUsageStats=false',
            '--theme.primaryColor=#667eea',
            '--theme.backgroundColor=#ffffff',
            '--theme.secondaryBackgroundColor=#f0f2f6',
            '--logger.level=info'
        ]
        
        logger.info("[INFO] Starting Streamlit server...")
        logger.info("[INFO] Access the app at: http://localhost:8501")
        logger.info("[INFO] The application will open automatically in your browser")
        
        # Run Streamlit in foreground so user can see it
        process = subprocess.run(cmd)
        return process.returncode == 0
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to launch Streamlit app: {e}")
        logger.info("[TIP] Try running manually: streamlit run app/pecunia_app.py")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Pecunia AI - Complete Financial Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete system (recommended)
  python main_production.py
  
  # Run with custom training settings
  python main_production.py --sample-size 50000 --epochs 30
  
  # Skip training and just run app
  python main_production.py --skip-training
  
  # Force retrain all models
  python main_production.py --force-retrain
  
  # Production deployment
  python main_production.py --production --install-deps
        """
    )
    
    # Training options
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Sample size for training (default: 10000)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
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
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    
    return parser.parse_args()

def print_banner():
    """Print the Pecunia AI banner"""
    banner = """
    ================================================================
    |                        PECUNIA AI                            |
    |              AI-Driven Financial Management                  |
    |                                                              |
    |  [NEWS] Smart Newsfeed       [EDU] Financial Education      |
    |  [FORUM] Community Forum     [AI] Analytics                 |  
    |  [ML] Models                 [INSIGHTS] Real-time Data      |
    ================================================================
    """
    print(banner)

async def main():
    """Main entry point for complete Pecunia AI system"""
    args = parse_arguments()
    
    print_banner()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("[DEBUG] Verbose mode enabled")
    
    logger.info("[START] Starting Pecunia AI Complete System...")
    
    try:
        # Step 1: Setup production environment if requested
        if args.production:
            setup_production_environment()
        
        # Step 2: Install dependencies if requested
        if args.install_deps:
            install_dependencies()
        
        # Step 3: Model training and detection
        if not args.skip_training:
            logger.info("[CHECK] Checking model status...")
            existing_models = detect_existing_models()
            
            if args.force_retrain:
                logger.info("[RETRAIN] Force retraining all models...")
                missing_models = ['fraud', 'expense', 'investment', 'income']
            else:
                missing_models = [name for name, exists in existing_models.items() if not exists]
            
            if missing_models:
                logger.info(f"[TRAIN] Need to train: {', '.join(missing_models)}")
                train_missing_models(missing_models, args.sample_size, args.epochs)
            else:
                logger.info("[OK] All models are available!")
        else:
            logger.info("[SKIP] Skipping model training as requested")
        
        # Step 4: Launch Streamlit application
        logger.info("[READY] All preparation complete - launching application...")
        time.sleep(2)  # Brief pause for visibility
        
        success = launch_streamlit_app()
        
        if success:
            logger.info("[SUCCESS] Pecunia AI launched successfully!")
        else:
            logger.error("[ERROR] Failed to launch Pecunia AI")
            
    except KeyboardInterrupt:
        logger.info("\n[STOP] Operation cancelled by user")
    except Exception as e:
        logger.error(f"\n[ERROR] Error in main execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    logger.info("\n[COMPLETE] Pecunia AI - Ready for Production!")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 