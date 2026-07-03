import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import split_data_by_airfoil
from src.solver import AirfoilSolver

def load_all_models():
    models_dir = "models/weights"
    
    models = {
        'Linear Regression': joblib.load(os.path.join(models_dir, "linear_regression.joblib")),
        'Polynomial Regression': joblib.load(os.path.join(models_dir, "polynomial_regression_deg_2.joblib")),
        'KNN Regressor': joblib.load(os.path.join(models_dir, "k-nearest_neighbors_knn.joblib")),
        'Decision Tree': joblib.load(os.path.join(models_dir, "decision_tree.joblib")),
        'Random Forest': joblib.load(os.path.join(models_dir, "random_forest.joblib")),
        'MLP Regressor': joblib.load(os.path.join(models_dir, "multi-layer_perceptron_mlp.joblib")),
    }
    return models

def evaluate_inference_speed(coords, Re, models, solver):
    print("\n=== Inference Speed Benchmark ===")
    alphas = np.linspace(-5.0, 15.0, 100) # 100 test points
    
    # 1. Physics solver time (single alpha point)
    start_time = time.perf_counter()
    for alpha in alphas:
        _ = solver.solve(coords, alpha, Re)
    solver_time = (time.perf_counter() - start_time) / len(alphas)
    print(f"Physics Solver average time per run: {solver_time * 1000:.3f} ms")
    
    # Extract features for ML
    from src.geometry import extract_geometric_features, fit_cst
    geom = extract_geometric_features(coords)
    w_up, w_lo = fit_cst(coords, order=5)
    
    # Tabular input vector
    row_list = []
    for alpha in alphas:
        row = {
            'alpha': alpha,
            'Re': Re,
            'max_thickness': geom['max_thickness'],
            'max_thickness_loc': geom['max_thickness_loc'],
            'max_camber': geom['max_camber'],
            'max_camber_loc': geom['max_camber_loc'],
            'le_radius': geom['le_radius']
        }
        for i, v in enumerate(w_up):
            row[f'cst_up_{i}'] = v
        for i, v in enumerate(w_lo):
            row[f'cst_lo_{i}'] = v
        row_list.append(row)
        
    df_input = pd.DataFrame(row_list)
    feature_cols = ['alpha', 'Re'] + list(geom.keys()) + [f'cst_up_{i}' for i in range(6)] + [f'cst_lo_{i}' for i in range(6)]
    X_input = df_input[feature_cols]
    
    # Run speedups for each scikit-learn model
    for name, pipeline in models.items():
        start_time = time.perf_counter()
        for _ in range(20):
            _ = pipeline.predict(X_input)
        model_time = (time.perf_counter() - start_time) / (20 * len(alphas))
        print(f"{name:<25} average time: {model_time * 1000:.5f} ms (Speedup: {solver_time / (model_time + 1e-9):.1f}x)")

def run_error_analysis(test_df, models):
    print("\n=== Error Analysis on Test Airfoils ===")
    
    # Feature names
    geom_cols = ['max_thickness', 'max_thickness_loc', 'max_camber', 'max_camber_loc', 'le_radius']
    cst_cols = [c for c in test_df.columns if c.startswith('cst_')]
    feature_cols = ['alpha', 'Re'] + geom_cols + cst_cols
    
    X_test = test_df[feature_cols]
    y_true = test_df[['CL', 'CD', 'Cm']].values
    
    # Get predictions
    preds = {}
    for name, pipeline in models.items():
        preds[name] = pipeline.predict(X_test)
        
    # Analyze errors across different subsets:
    # 1. Alpha ranges: pre-stall (< 10) vs post-stall (>= 10)
    pre_stall_idx = test_df['alpha'] < 10.0
    post_stall_idx = test_df['alpha'] >= 10.0
    
    # 2. Thickness: thin (<10%) vs thick (>=10%)
    thin_idx = test_df['max_thickness'] < 0.10
    thick_idx = test_df['max_thickness'] >= 0.10
    
    subsets = {
        'All Test Conditions': slice(None),
        'Pre-stall (Alpha < 10)': pre_stall_idx,
        'Post-stall (Alpha >= 10)': post_stall_idx,
        'Thin Airfoils (<10% t/c)': thin_idx,
        'Thick Airfoils (>=10% t/c)': thick_idx
    }
    
    # Print results in a markdown table format
    print("| Subset | Model | CL MAE | CD MAE | Cm MAE |")
    print("|---|---|---|---|---|")
    
    for sub_name, mask in subsets.items():
        if np.sum(mask) == 0:
            continue
            
        y_true_sub = y_true[mask]
        
        for model_name, y_pred in preds.items():
            y_pred_sub = y_pred[mask]
            mae_cl = mean_absolute_error(y_true_sub[:, 0], y_pred_sub[:, 0])
            mae_cd = mean_absolute_error(y_true_sub[:, 1], y_pred_sub[:, 1])
            mae_cm = mean_absolute_error(y_true_sub[:, 2], y_pred_sub[:, 2])
            print(f"| {sub_name} | {model_name} | {mae_cl:.4f} | {mae_cd:.4f} | {mae_cm:.4f} |")

def plot_polar_curves(test_df, models, solver):
    print("\n=== Generating Polar Curve Visualizations ===")
    test_airfoils = test_df['airfoil_name'].unique()
    if len(test_airfoils) == 0:
        return
        
    airfoil_name = test_airfoils[0]
    print(f"Generating polars for test airfoil: {airfoil_name}")
    
    coords_dict = np.load("data/processed/airfoil_coords.npz")
    coords = coords_dict[airfoil_name]
    
    from src.geometry import extract_geometric_features, fit_cst
    geom = extract_geometric_features(coords)
    w_up, w_lo = fit_cst(coords, order=5)
    
    alphas = np.arange(-5.0, 16.0, 1.0)
    Re = 1e6
    
    # Ground Truth Solver polars
    solver_results = [solver.solve(coords, a, Re) for a in alphas]
    cl_gt = [r['CL'] for r in solver_results]
    cd_gt = [r['CD'] for r in solver_results]
    cm_gt = [r['Cm'] for r in solver_results]
    
    # Construct input dataframe for evaluation
    rows = []
    for a in alphas:
        row = {
            'alpha': a,
            'Re': Re,
            'max_thickness': geom['max_thickness'],
            'max_thickness_loc': geom['max_thickness_loc'],
            'max_camber': geom['max_camber'],
            'max_camber_loc': geom['max_camber_loc'],
            'le_radius': geom['le_radius']
        }
        for i, v in enumerate(w_up):
            row[f'cst_up_{i}'] = v
        for i, v in enumerate(w_lo):
            row[f'cst_lo_{i}'] = v
        rows.append(row)
        
    df_eval = pd.DataFrame(rows)
    feature_cols = ['alpha', 'Re'] + list(geom.keys()) + [f'cst_up_{i}' for i in range(6)] + [f'cst_lo_{i}' for i in range(6)]
    
    # Predict curves
    preds_curves = {}
    for name, pipeline in models.items():
        preds_curves[name] = pipeline.predict(df_eval[feature_cols])
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Set custom colors
    colors = {
        'Linear Regression': '#e11d48',
        'Polynomial Regression': '#ea580c',
        'KNN Regressor': '#16a34a',
        'Decision Tree': '#2563eb',
        'Random Forest': '#9333ea',
        'MLP Regressor': '#0ea5e9'
    }
    
    # CL vs Alpha
    axes[0].plot(alphas, cl_gt, 'k-', linewidth=3, label='Solver (GT)')
    for name, pred in preds_curves.items():
        axes[0].plot(alphas, pred[:, 0], '--', color=colors[name], label=name)
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Lift Coefficient (CL)')
    axes[0].set_title('Lift Polar Curve')
    axes[0].grid(True)
    axes[0].legend(fontsize=9)
    
    # CD vs Alpha
    axes[1].plot(alphas, cd_gt, 'k-', linewidth=3, label='Solver (GT)')
    for name, pred in preds_curves.items():
        axes[1].plot(alphas, pred[:, 1], '--', color=colors[name], label=name)
    axes[1].set_xlabel('Angle of Attack (deg)')
    axes[1].set_ylabel('Drag Coefficient (CD)')
    axes[1].set_title('Drag Polar Curve')
    axes[1].grid(True)
    
    # Cm vs Alpha
    axes[2].plot(alphas, cm_gt, 'k-', linewidth=3, label='Solver (GT)')
    for name, pred in preds_curves.items():
        axes[2].plot(alphas, pred[:, 2], '--', color=colors[name], label=name)
    axes[2].set_xlabel('Angle of Attack (deg)')
    axes[2].set_ylabel('Pitching Moment (Cm)')
    axes[2].set_title('Pitching Moment Polar Curve')
    axes[2].grid(True)
    
    plt.suptitle(f"Aerodynamic Polar Predictions Comparison for Unseen Airfoil: {airfoil_name} at Re = {Re:,.0f}", fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save plot
    os.makedirs("data/processed", exist_ok=True)
    plot_path = "data/processed/polar_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved polar comparison plot to {plot_path}")
    plt.close()

def main():
    dataset_path = "data/processed/airfoil_dataset.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Run generator.py first!")
        return
        
    df = pd.read_csv(dataset_path)
    _, test_df = split_data_by_airfoil(df)
    
    print("Loading models for evaluation...")
    models = load_all_models()
    
    solver = AirfoilSolver()
    
    # Inference Speed Benchmark
    coords_dict = np.load("data/processed/airfoil_coords.npz")
    airfoil_name = test_df['airfoil_name'].iloc[0]
    coords = coords_dict[airfoil_name]
    evaluate_inference_speed(coords, 1e6, models, solver)
    
    # Error analysis
    run_error_analysis(test_df, models)
    
    # Plotting
    plot_polar_curves(test_df, models, solver)

if __name__ == "__main__":
    main()
