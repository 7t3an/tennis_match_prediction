#!/usr/bin/env python3
"""
Tennis Match Prediction - Automated Pipeline Script.

This script handles:
1. Data update from TML repository (git pull) - only new data
2. Data processing and feature engineering
3. Model training with proper validation
4. Model saving for Streamlit app

Usage:
    # Full pipeline (update data + retrain model)
    python scripts/run_pipeline.py --full
    
    # Production mode (train on ALL data 2012-2025)
    python scripts/run_pipeline.py --full --production
    
    # Only retrain model (no data update)
    python scripts/run_pipeline.py --train
    
    # Only update data
    python scripts/run_pipeline.py --update-data
    
    # Quick test (smaller dataset)
    python scripts/run_pipeline.py --test
"""

import argparse
import subprocess
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import TennisPredictionPipeline


def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file to detect changes."""
    if not filepath.exists():
        return ""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def update_data_from_git():
    """Update tennis data from TML GitHub repository.
    
    Uses git pull which only downloads new/changed files.
    Safe to call multiple times - won't duplicate data.
    """
    print("=" * 80)
    print("UPDATING DATA FROM GIT REPOSITORY")
    print("=" * 80)
    
    tml_path = project_root / 'data' / 'tml'
    
    if not tml_path.exists():
        print(f"Error: TML data directory not found: {tml_path}")
        return False
    
    # Store hash of 2025.csv before update
    csv_2025 = tml_path / '2025.csv'
    hash_before = get_file_hash(csv_2025)
    
    try:
        # Check if it's a git repository
        result = subprocess.run(
            ['git', 'status'],
            cwd=tml_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("Note: TML directory is not a git repository")
            print("Data will be used as-is")
            return True
        
        # Pull latest changes (git only downloads what's new)
        print("Pulling latest data...")
        result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            cwd=tml_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Try master branch
            result = subprocess.run(
                ['git', 'pull', 'origin', 'master'],
                cwd=tml_path,
                capture_output=True,
                text=True
            )
        
        if result.returncode == 0:
            print(result.stdout)
            
            # Check if 2025.csv changed
            hash_after = get_file_hash(csv_2025)
            if hash_before != hash_after:
                print("✅ New data detected in 2025.csv!")
            else:
                print("ℹ️  No new data available (already up to date)")
            
            return True
        else:
            print(f"Warning: Could not pull data: {result.stderr}")
            return True  # Continue anyway
                
    except FileNotFoundError:
        print("Git not found - skipping data update")
        return True


def run_full_pipeline(start_year: int = 2012, end_year: int = 2025, test_year: int = 2025, production: bool = False):
    """Run the complete ML pipeline.
    
    Args:
        start_year: First year of data to use
        end_year: Last year of data to use
        test_year: Year to use for testing (if not production mode)
        production: If True, train on ALL data (no test split) for deployment
    """
    if production:
        print("\n" + "=" * 80)
        print("RUNNING PRODUCTION PIPELINE (TRAIN ON ALL DATA)")
        print(f"Period: {start_year}-{end_year}")
        print("⚠️  Production mode: no test split, training on complete dataset")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("RUNNING FULL ML PIPELINE")
        print(f"Period: {start_year}-{end_year}, Test: {test_year}")
        print("=" * 80)
    
    pipeline = TennisPredictionPipeline(
        data_path='data/tml',
        output_path='data/processed',
        model_path='models'
    )
    
    if production:
        # Production mode: train on ALL data
        results = pipeline.run_production(
            start_year=start_year,
            end_year=end_year
        )
    else:
        # Development mode: keep test split
        results = pipeline.run(
            start_year=start_year,
            end_year=end_year,
            test_year=test_year,
            run_cv=True,
            save_data=True,
            save_model=True
        )
    
    return results


def retrain_on_full_data():
    """Retrain model on complete dataset after validation."""
    print("\n" + "=" * 80)
    print("RETRAINING ON FULL DATA (2012-2025)")
    print("=" * 80)
    
    pipeline = TennisPredictionPipeline()
    
    # First run validation
    results = pipeline.run(
        start_year=2012,
        end_year=2025,
        test_year=2025,
        run_cv=True,
        save_data=True,
        save_model=False  # Don't save yet
    )
    
    # Check if metrics are acceptable
    if results.get('test_metrics'):
        roc_auc = results['test_metrics'].get('roc_auc', 0)
        
        if roc_auc < 0.60:
            print(f"\n⚠️ WARNING: ROC-AUC ({roc_auc:.4f}) is below 0.60")
            print("Consider investigating data or features before retraining!")
            
            response = input("Continue with retraining anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborting retrain")
                return None
    
    # Retrain on full data
    retrain_results = pipeline.retrain_on_full_data()
    
    return retrain_results


def main():
    parser = argparse.ArgumentParser(
        description='Tennis Match Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py --full        # Complete pipeline
    python scripts/run_pipeline.py --train       # Only train model
    python scripts/run_pipeline.py --update-data # Only update data
    python scripts/run_pipeline.py --test        # Quick test
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full pipeline (update data + train model)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train model (no data update)'
    )
    
    parser.add_argument(
        '--update-data',
        action='store_true',
        help='Only update data from git'
    )
    
    parser.add_argument(
        '--retrain-full',
        action='store_true',
        help='Retrain model on complete dataset (after validation)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Quick test with smaller dataset (2020-2025)'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Production mode: train on ALL data (2012-2025) without test split'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        default=2012,
        help='Start year for data (default: 2012)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        default=2025,
        help='End year for data (default: 2025)'
    )
    
    parser.add_argument(
        '--test-year',
        type=int,
        default=2025,
        help='Year to use as test set (default: 2025)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("TENNIS MATCH PREDICTION PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Default to full if no arguments
    if not any([args.full, args.train, args.update_data, args.test, args.retrain_full]):
        args.full = True
    
    # Update data if requested
    if args.full or args.update_data:
        success = update_data_from_git()
        if not success:
            print("Data update failed!")
            sys.exit(1)
        
        if args.update_data and not (args.full or args.train):
            print("\nData update complete!")
            sys.exit(0)
    
    # Quick test mode
    if args.test:
        results = run_full_pipeline(
            start_year=2020,
            end_year=2025,
            test_year=2025,
            production=False
        )
    # Retrain on full data
    elif args.retrain_full:
        results = retrain_on_full_data()
    # Full or train mode
    elif args.full or args.train:
        results = run_full_pipeline(
            start_year=args.start_year,
            end_year=args.end_year,
            test_year=args.test_year,
            production=args.production
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if results and 'test_metrics' in results:
        metrics = results['test_metrics']
        print(f"\n📊 Final Metrics:")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Log Loss:  {metrics['log_loss']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        
        # Check targets
        print("\n🎯 Target Status:")
        if metrics['roc_auc'] >= 0.70:
            print("   ✅ ROC-AUC >= 0.70 - ACHIEVED!")
        else:
            print(f"   ⚠️ ROC-AUC = {metrics['roc_auc']:.4f} (target: 0.70)")


if __name__ == '__main__':
    main()
