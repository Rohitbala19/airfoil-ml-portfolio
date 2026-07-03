import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    get_linear_model,
    get_polynomial_model,
    get_knn_model,
    get_decision_tree,
    get_random_forest,
    get_mlp_model
)

# Set random seed for reproducibility
np.random.seed(42)

# --- Train/Val/Test Split by Airfoil Shape ---
def split_data_by_airfoil(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Rigorously splits data by Airfoil Profile.
    This guarantees that the test set evaluates generalization to entirely unseen geometries,
    preventing data leakage.
    """
    airfoils = list(df['airfoil_name'].unique())
    train_names, test_names = train_test_split(airfoils, test_size=(val_ratio + test_ratio), random_state=42)
    val_names, test_names = train_test_split(test_names, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    
    print(f"Airfoil Splitting:")
    print(f"  Train Airfoils ({len(train_names)}): {list(train_names)}")
    print(f"  Val Airfoils ({len(val_names)}): {list(val_names)}")
    print(f"  Test Airfoils ({len(test_names)}): {list(test_names)}")
    
    train_df = df[df['airfoil_name'].isin(train_names)].copy()
    val_df = df[df['airfoil_name'].isin(val_names)].copy()
    test_df = df[df['airfoil_name'].isin(test_names)].copy()
    
    # Standardize train and val together for scikit-learn models (combine train + val as training set)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    
    return train_val_df, test_df

def main():
    dataset_path = "data/processed/airfoil_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Run generator.py first!")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Define features and targets
    geom_cols = ['max_thickness', 'max_thickness_loc', 'max_camber', 'max_camber_loc', 'le_radius']
    cst_cols = [c for c in df.columns if c.startswith('cst_')]
    feature_cols = ['alpha', 'Re'] + geom_cols + cst_cols
    target_cols = ['CL', 'CD', 'Cm']
    
    # Split datasets
    train_df, test_df = split_data_by_airfoil(df)
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    # Instantiate models
    # We wrap models in a Pipeline with StandardScaler to automate feature scaling
    models = {
        'Linear Regression': make_pipeline(StandardScaler(), get_linear_model()),
        'Polynomial Regression (deg 2)': get_polynomial_model(degree=2), # PolynomialFeatures handles scaling internally
        'K-Nearest Neighbors (KNN)': make_pipeline(StandardScaler(), get_knn_model()),
        'Decision Tree': get_decision_tree(max_depth=10),
        'Random Forest': get_random_forest(n_estimators=100, max_depth=12),
        'Multi-Layer Perceptron (MLP)': make_pipeline(StandardScaler(), get_mlp_model())
    }
    
    os.makedirs("models/weights", exist_ok=True)
    
    # Training Loop
    print("\n=== Training Models ===")
    results = {}
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        
        # Save trained pipeline
        model_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, f"models/weights/{model_filename}.joblib")
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        results[name] = y_pred
        
    # Evaluate Models on Unseen Airfoil geometries
    print("\n=== Model Evaluation (Generalization on Unseen Airfoils) ===")
    for name, pred in results.items():
        print(f"\n📈 {name}:")
        for i, col in enumerate(target_cols):
            mse = mean_squared_error(y_test.values[:, i], pred[:, i])
            mae = mean_absolute_error(y_test.values[:, i], pred[:, i])
            r2 = r2_score(y_test.values[:, i], pred[:, i])
            print(f"  {col} -> MSE: {mse:.6f}, MAE: {mae:.5f}, R2: {r2:.4f}")
            
    print("\nTraining complete! All models saved to models/weights/")

if __name__ == "__main__":
    main()
