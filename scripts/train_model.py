#!/usr/bin/env python3
"""
Model training script.

Train: 2012-2024
Test: 2025-2026
Final: 2012-2026 (production model)
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import TennisPredictionPipeline
import pandas as pd


def main():
    print("=" * 80)
    print("TENNIS MATCH PREDICTION - MODEL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    pipeline = TennisPredictionPipeline(
        data_path='tml-data',
        output_path='data/processed',
        model_path='models'
    )
    
    # Step 1: Train on 2012-2024, Test on 2025-2026
    print("\nStep 1: Training and Testing")
    print("Train: 2012-2024 | Test: 2025-2026")
    
    results = pipeline.run(
        start_year=2012,
        end_year=2026,
        test_year=2025,
        run_cv=True,
        save_data=True,
        save_model=False
    )
    
    if 'error' in results:
        print(f"\nError: {results['error']}")
        sys.exit(1)
    
    # Display metrics
    metrics = results['test_metrics']
    print("\nTest Metrics (2025-2026):")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Log Loss:  {metrics['log_loss']:.4f}")
    
    # Step 2: Retrain on all data
    print("\nStep 2: Retraining on Full Data (2012-2026)")
    
    pipeline.retrain_on_full_data()
    pipeline._save_metrics(metrics, 2012, 2026, 2025)
    
    # Create full_features.csv
    print("\nCreating full_features.csv...")
    train_features = pd.read_csv('data/processed/train_features.csv')
    test_features = pd.read_csv('data/processed/test_features.csv')
    full_features = pd.concat([train_features, test_features], ignore_index=True)
    full_features.to_csv('data/processed/full_features.csv', index=False)
    print(f"Saved: data/processed/full_features.csv ({len(full_features):,} matches)")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\nSaved files:")
    print("  Data:  data/processed/{train,test,full}_features.csv")
    print("  Model: models/xgboost_calibrated_model.pkl")
    print("\nRun: streamlit run app.py")


if __name__ == '__main__':
    main()
