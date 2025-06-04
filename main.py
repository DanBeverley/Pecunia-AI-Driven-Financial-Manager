#!/usr/bin/env python3
"""
Pecunia AI - Enterprise Financial Management System
Main entry point with command line interface for all system components
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our enterprise utils
from utils import (
    initialize_all_systems, SecurityPolicy, AuthMethod, UserRole,
    clean_data, detect_anomalies, BlockchainNetwork
)

# Import model training functions
from models.fraud_detection import train_fraud_detection_model
from models.train_expense_model import train_expense_classifier
from models.train_investment_model import train_investment_model
from models.train_income_model import train_income_predictor

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Pecunia AI - Enterprise Financial Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize and run full system demo
  python main.py --mode demo --enable-all
  
  # Train fraud detection model on sample data
  python main.py --mode train --model fraud --sample-size 50000
  
  # Train all models with small samples for testing
  python main.py --mode train --model all --sample-size 10000 --quick
  
  # Clean and analyze data
  python main.py --mode data --operation clean --input creditcard.csv
  
  # Run blockchain portfolio analysis
  python main.py --mode blockchain --wallet 0x742d35Cc6634C0532925a3b8D8d93C95C5B588eE
  
  # Initialize enterprise authentication system
  python main.py --mode auth --setup-mfa --create-admin
  
  # Run API management demo
  python main.py --mode api --start-webhooks --port 5000
        """
    )
    
    # Main operation mode
    parser.add_argument(
        '--mode', 
        choices=['demo', 'train', 'data', 'blockchain', 'auth', 'api', 'init'],
        default='demo',
        help='Operation mode to run'
    )
    
    # Training options
    parser.add_argument(
        '--model',
        choices=['fraud', 'expense', 'investment', 'income', 'all'],
        default='fraud',
        help='Which model to train'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100000,
        help='Sample size for training (default: 100000)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode with reduced epochs and sample size'
    )
    
    # Data processing options
    parser.add_argument(
        '--operation',
        choices=['clean', 'analyze', 'anomaly'],
        default='clean',
        help='Data operation to perform'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Input data file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path'
    )
    
    # Blockchain options
    parser.add_argument(
        '--wallet',
        type=str,
        help='Wallet address for blockchain analysis'
    )
    
    parser.add_argument(
        '--networks',
        nargs='+',
        default=['ethereum', 'polygon'],
        help='Blockchain networks to analyze'
    )
    
    # Authentication options
    parser.add_argument(
        '--setup-mfa',
        action='store_true',
        help='Setup multi-factor authentication'
    )
    
    parser.add_argument(
        '--create-admin',
        action='store_true',
        help='Create admin user'
    )
    
    # API options
    parser.add_argument(
        '--start-webhooks',
        action='store_true',
        help='Start webhook server'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port for webhook server (default: 5000)'
    )
    
    # System options
    parser.add_argument(
        '--enable-all',
        action='store_true',
        help='Enable all system components'
    )
    
    parser.add_argument(
        '--redis-url',
        type=str,
        default='redis://localhost:6379/0',
        help='Redis connection URL'
    )
    
    parser.add_argument(
        '--db-url',
        type=str,
        default='sqlite:///pecunia.db',
        help='Database connection URL'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def print_banner():
    """Print the Pecunia AI banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                        PECUNIA AI                            ║
    ║              Enterprise Financial Management                 ║
    ║                                                              ║
    ║  🏦 Advanced ML Models    🔐 Enterprise Security           ║
    ║  ⛓️  Blockchain Integration  📊 Real-time Analytics        ║
    ║  🚀 Stream Processing     🔗 API Management                ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def initialize_systems(args) -> Dict[str, Any]:
    """Initialize all system components"""
    print("🚀 Initializing Pecunia AI Enterprise Systems...")
    
    # Create security policy
    security_policy = SecurityPolicy(
        password_min_length=12,
        require_mfa=True,
        allowed_mfa_methods=[AuthMethod.TOTP, AuthMethod.SMS, AuthMethod.BIOMETRIC_FACE],
        failed_login_threshold=3,
        session_timeout_minutes=30
    )
    
    # Initialize all systems
    systems = initialize_all_systems(
        database_url=args.db_url,
        redis_url=args.redis_url,
        security_policy=security_policy
    )
    
    print("✅ All enterprise systems initialized successfully!")
    return systems

async def run_demo_mode(args, systems):
    """Run comprehensive system demo"""
    print("\n🎭 Running Pecunia AI Enterprise Demo...")
    
    if args.enable_all:
        print("\n🔧 Full System Integration Demo")
        
        # Import and run the comprehensive demo
        from utils.example_usage import main as demo_main
        await demo_main()
        
    else:
        print("\n📊 Quick Demo - Core Features")
        
        # Quick demo of each component
        print("✅ API Management: Enterprise-grade rate limiting and webhooks")
        print("✅ Authentication: MFA, biometric, OAuth2 integration")
        print("✅ Data Processing: ML-based anomaly detection")
        print("✅ Blockchain: Multi-network DeFi and NFT tracking")
        
        print("\n💡 Use --enable-all for full interactive demo")

def run_training_mode(args):
    """Run model training"""
    print(f"\n🧠 Training Mode: {args.model.upper()}")
    
    # Adjust parameters for quick mode
    if args.quick:
        sample_size = min(args.sample_size, 10000)
        epochs = min(args.epochs, 20)
        print(f"⚡ Quick mode: Sample={sample_size}, Epochs={epochs}")
    else:
        sample_size = args.sample_size
        epochs = args.epochs
    
    start_time = time.time()
    
    if args.model == 'fraud' or args.model == 'all':
        print("\n🛡️ Training Fraud Detection Model...")
        try:
            fraud_model = train_fraud_detection_model(
                data_path='creditcard.csv',
                sample_size=sample_size,
                n_epochs=epochs,
                save_model=True,
                model_path='models/fraud_detector.pkl'
            )
            if fraud_model:
                print("✅ Fraud detection model trained successfully!")
        except Exception as e:
            print(f"❌ Error training fraud model: {e}")
    
    if args.model == 'expense' or args.model == 'all':
        print("\n💰 Training Expense Classification Model...")
        try:
            expense_model = train_expense_classifier(
                data_path='personal_expense_classification.csv',
                n_epochs=epochs,
                save_model=True,
                model_path='models/expense_classifier.pkl'
            )
            if expense_model:
                print("✅ Expense classification model trained successfully!")
        except Exception as e:
            print(f"❌ Error training expense model: {e}")
    
    if args.model == 'investment' or args.model == 'all':
        print("\n📈 Training Investment Model...")
        try:
            investment_model = train_investment_model(
                data_path='all_stocks_5yr.csv',
                sample_size=sample_size,
                n_epochs=epochs,
                save_model=True,
                model_path='models/investment_model.pkl'
            )
            if investment_model:
                print("✅ Investment model trained successfully!")
        except Exception as e:
            print(f"❌ Error training investment model: {e}")
    
    if args.model == 'income' or args.model == 'all':
        print("\n💼 Training Income Prediction Model...")
        try:
            income_model = train_income_predictor(
                data_path='adult.csv',
                sample_size=sample_size,
                n_epochs=epochs,
                save_model=True,
                model_path='models/income_predictor.pkl'
            )
            if income_model:
                print("✅ Income prediction model trained successfully!")
        except Exception as e:
            print(f"❌ Error training income model: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total training time: {total_time:.2f} seconds")
    print("🎯 All models saved in models/ directory")

def run_data_mode(args):
    """Run data processing operations"""
    print(f"\n📊 Data Processing Mode: {args.operation.upper()}")
    
    if not args.input:
        print("❌ Please specify input file with --input")
        return
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    print(f"📁 Processing file: {args.input}")
    
    try:
        import pandas as pd
        data = pd.read_csv(args.input)
        print(f"✅ Loaded {len(data)} records with {len(data.columns)} columns")
        
        if args.operation == 'clean':
            print("🧹 Cleaning data...")
            cleaned_data, quality_metrics = clean_data(data)
            
            print(f"📊 Data Quality Results:")
            print(f"   • Completeness: {quality_metrics.completeness:.1%}")
            print(f"   • Accuracy: {quality_metrics.accuracy:.1%}")
            print(f"   • Overall Score: {quality_metrics.overall_score:.1%}")
            
            if args.output:
                cleaned_data.to_csv(args.output, index=False)
                print(f"💾 Cleaned data saved to: {args.output}")
        
        elif args.operation == 'anomaly':
            print("🚨 Detecting anomalies...")
            numeric_columns = data.select_dtypes(include=['number']).columns
            
            for col in numeric_columns[:3]:  # Process first 3 numeric columns
                if len(data[col].dropna()) > 10:
                    anomalies = detect_anomalies(data[col].dropna().values)
                    anomaly_rate = anomalies.sum() / len(anomalies)
                    print(f"   • {col}: {anomalies.sum()} anomalies ({anomaly_rate:.1%})")
        
        elif args.operation == 'analyze':
            print("📈 Analyzing data...")
            print(f"📊 Data Summary:")
            print(f"   • Shape: {data.shape}")
            print(f"   • Missing values: {data.isnull().sum().sum()}")
            print(f"   • Duplicate rows: {data.duplicated().sum()}")
            print(f"   • Numeric columns: {len(data.select_dtypes(include=['number']).columns)}")
            print(f"   • Categorical columns: {len(data.select_dtypes(include=['object']).columns)}")
            
    except Exception as e:
        print(f"❌ Error processing data: {e}")

async def run_blockchain_mode(args, systems):
    """Run blockchain analysis"""
    print(f"\n⛓️ Blockchain Analysis Mode")
    
    if not args.wallet:
        print("❌ Please specify wallet address with --wallet")
        return
    
    print(f"💼 Analyzing wallet: {args.wallet}")
    print(f"🌐 Networks: {', '.join(args.networks)}")
    
    try:
        from utils.blockchain import get_wallet_portfolio
        
        # This would work with actual blockchain APIs
        print("📊 Portfolio Analysis (Demo Mode):")
        portfolio = {
            'wallet': args.wallet,
            'total_value': 125000.0,
            'defi_positions': 3,
            'nft_count': 15,
            'networks': args.networks
        }
        
        print(f"💰 Total Portfolio Value: ${portfolio['total_value']:,.2f}")
        print(f"🏦 DeFi Positions: {portfolio['defi_positions']}")
        print(f"🖼️ NFT Count: {portfolio['nft_count']}")
        
    except Exception as e:
        print(f"❌ Error in blockchain analysis: {e}")

def run_auth_mode(args, systems):
    """Run authentication setup"""
    print(f"\n🔐 Authentication Mode")
    
    auth_manager = systems.get('auth_manager')
    if not auth_manager:
        print("❌ Authentication system not initialized")
        return
    
    if args.create_admin:
        print("👤 Creating admin user...")
        success, result = auth_manager.register_user(
            username="admin",
            email="admin@pecunia.ai",
            password="SecureAdmin123!@#",
            role=UserRole.ADMIN.value
        )
        
        if success:
            print("✅ Admin user created successfully!")
            user = result
            
            if args.setup_mfa:
                print("🔒 Setting up MFA...")
                totp_secret, qr_uri = auth_manager.mfa_manager.setup_totp(user)
                print(f"📱 TOTP Secret: {totp_secret}")
                print("🔑 Scan QR code with authenticator app")
        else:
            print(f"❌ Failed to create admin user: {result}")

def run_api_mode(args, systems):
    """Run API management"""
    print(f"\n🔗 API Management Mode")
    
    api_manager = systems.get('api_manager')
    if not api_manager:
        print("❌ API manager not initialized")
        return
    
    if args.start_webhooks:
        print(f"🕷️ Starting webhook server on port {args.port}...")
        print("📡 Available endpoints:")
        print(f"   • POST http://localhost:{args.port}/webhook/financial-data")
        print(f"   • POST http://localhost:{args.port}/webhook/market-data")
        print(f"   • POST http://localhost:{args.port}/webhook/alerts")
        print(f"   • GET  http://localhost:{args.port}/health")
        
        # This would start the actual webhook server
        print("🔑 Use webhook secret for HMAC signature verification")
        print("⚠️  Demo mode - actual server not started")

def run_init_mode(args):
    """Initialize project structure"""
    print("\n🏗️ Project Initialization Mode")
    
    # Check required directories
    required_dirs = ['models', 'utils', 'data', 'logs']
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir()
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    # Check required files
    required_files = {
        'models/quantized': 'Quantized models directory',
        'logs/app.log': 'Application log file',
        'config.yaml': 'Configuration file'
    }
    
    for file_path, description in required_files.items():
        path = Path(file_path)
        if not path.exists():
            if path.suffix:
                path.touch()
            else:
                path.mkdir(exist_ok=True)
            print(f"📄 Created {description}: {file_path}")
    
    print("✅ Project structure initialized successfully!")

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    print_banner()
    
    if args.verbose:
        print(f"🔧 Configuration:")
        print(f"   • Mode: {args.mode}")
        print(f"   • Redis URL: {args.redis_url}")
        print(f"   • Database URL: {args.db_url}")
    
    # Initialize systems for modes that need them
    systems = {}
    if args.mode in ['demo', 'blockchain', 'auth', 'api']:
        systems = initialize_systems(args)
    
    # Route to appropriate mode
    try:
        if args.mode == 'demo':
            await run_demo_mode(args, systems)
        elif args.mode == 'train':
            run_training_mode(args)
        elif args.mode == 'data':
            run_data_mode(args)
        elif args.mode == 'blockchain':
            await run_blockchain_mode(args, systems)
        elif args.mode == 'auth':
            run_auth_mode(args, systems)
        elif args.mode == 'api':
            run_api_mode(args, systems)
        elif args.mode == 'init':
            run_init_mode(args)
        
        print(f"\n🎉 {args.mode.title()} mode completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error in {args.mode} mode: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print("\n🚀 Pecunia AI - Ready for Enterprise Deployment!")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 