import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AirfoilMLP, AirfoilCNN1D
from src.train import split_data_by_airfoil
from src.solver import AirfoilSolver

def load_all_models(cst_dim=12, device='cpu'):
    # Load baselines
    lr = joblib.load("models/weights/linear_regression.joblib")
    
    xgb_models = {}
    for col in ['CL', 'CD', 'Cm']:
        xgb_models[col] = joblib.load(f"models/weights/xgb_{col}.joblib")
        
    # Load PyTorch models
    mlp = AirfoilMLP(cst_dim=cst_dim)
    mlp.load_state_dict(torch.load("models/weights/airfoilmlp.pth", map_location=device))
    mlp.eval()
    
    cnn = AirfoilCNN1D(coord_len=200)
    cnn.load_state_dict(torch.load("models/weights/airfoilcnn1d.pth", map_location=device))
    cnn.eval()
    
    return lr, xgb_models, mlp, cnn

def evaluate_inference_speed(coords, Re, lr, xgb_models, mlp, cnn, solver):
    print("\n=== Inference Speed Benchmark ===")
    alphas = np.linspace(-5.0, 15.0, 100) # 100 test points
    
    # 1. Physics solver time (single alpha point)
    start_time = time.time()
    for alpha in alphas:
        _ = solver.solve(coords, alpha, Re)
    solver_time = (time.time() - start_time) / len(alphas)
    print(f"Physics Solver average time per run: {solver_time * 1000:.3f} ms")
    
    # Extract features for ML
    from src.geometry import extract_geometric_features, fit_cst
    geom = extract_geometric_features(coords)
    w_up, w_lo = fit_cst(coords, order=5)
    cst = np.concatenate([w_up, w_lo])
    
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
    
    # 2. Linear Regression time
    start_time = time.time()
    for _ in range(10):
        _ = lr.predict(X_input)
    lr_time = (time.time() - start_time) / (10 * len(alphas))
    print(f"Linear Regression average time:      {lr_time * 1000:.5f} ms (Speedup: {solver_time / (lr_time + 1e-9):.1f}x)")
    
    # 3. XGBoost time
    start_time = time.time()
    for _ in range(10):
        _ = xgb_models['CL'].predict(X_input)
        _ = xgb_models['CD'].predict(X_input)
        _ = xgb_models['Cm'].predict(X_input)
    xgb_time = (time.time() - start_time) / (10 * len(alphas))
    print(f"XGBoost average time:                {xgb_time * 1000:.5f} ms (Speedup: {solver_time / (xgb_time + 1e-9):.1f}x)")
    
    # 4. PyTorch MLP time
    cst_t = torch.tensor(X_input[[c for c in X_input.columns if c.startswith('cst_')]].values, dtype=torch.float32)
    alpha_t = torch.tensor(X_input['alpha'].values, dtype=torch.float32).unsqueeze(1)
    re_t = torch.tensor(X_input['Re'].values, dtype=torch.float32).unsqueeze(1)
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = mlp(cst_t, alpha_t, re_t)
    mlp_time = (time.time() - start_time) / (10 * len(alphas))
    print(f"PyTorch MLP average time:            {mlp_time * 1000:.5f} ms (Speedup: {solver_time / (mlp_time + 1e-9):.1f}x)")
    
    # 5. PyTorch CNN time
    coords_t = torch.tensor(coords.T, dtype=torch.float32).unsqueeze(0).repeat(len(alphas), 1, 1)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = cnn(coords_t, alpha_t, re_t)
    cnn_time = (time.time() - start_time) / (10 * len(alphas))
    print(f"PyTorch CNN 1D average time:         {cnn_time * 1000:.5f} ms (Speedup: {solver_time / (cnn_time + 1e-9):.1f}x)")

def run_error_analysis(test_df, lr, xgb_models, mlp, cnn):
    print("\n=== Error Analysis on Test Airfoils ===")
    
    # Feature names
    geom_cols = ['max_thickness', 'max_thickness_loc', 'max_camber', 'max_camber_loc', 'le_radius']
    cst_cols = [c for c in test_df.columns if c.startswith('cst_')]
    feature_cols = ['alpha', 'Re'] + geom_cols + cst_cols
    
    X_test = test_df[feature_cols]
    y_true = test_df[['CL', 'CD', 'Cm']].values
    
    # Run predictions
    pred_xgb = np.zeros_like(y_true)
    pred_xgb[:, 0] = xgb_models['CL'].predict(X_test)
    pred_xgb[:, 1] = xgb_models['CD'].predict(X_test)
    pred_xgb[:, 2] = xgb_models['Cm'].predict(X_test)
    
    # Convert PyTorch models to cpu
    mlp.cpu()
    cnn.cpu()
    
    cst_t = torch.tensor(X_test[cst_cols].values, dtype=torch.float32)
    alpha_t = torch.tensor(X_test['alpha'].values, dtype=torch.float32).unsqueeze(1)
    re_t = torch.tensor(X_test['Re'].values, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        pred_mlp = mlp(cst_t, alpha_t, re_t).numpy()
        
    coords_dict = np.load("data/processed/airfoil_coords.npz")
    coords_list = []
    for name in test_df['airfoil_name'].values:
        coords_list.append(torch.tensor(coords_dict[name].T, dtype=torch.float32))
    coords_t = torch.stack(coords_list)
    
    with torch.no_grad():
        pred_cnn = cnn(coords_t, alpha_t, re_t).numpy()
        
    # Analyze errors across different subsets:
    # 1. Alpha ranges: pre-stall (-5 to 9) vs post-stall (10 to 15)
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
        preds = {
            'XGBoost': pred_xgb[mask],
            'MLP': pred_mlp[mask],
            'CNN 1D': pred_cnn[mask]
        }
        
        for model_name, y_pred_sub in preds.items():
            mae_cl = mean_absolute_error(y_true_sub[:, 0], y_pred_sub[:, 0])
            mae_cd = mean_absolute_error(y_true_sub[:, 1], y_pred_sub[:, 1])
            mae_cm = mean_absolute_error(y_true_sub[:, 2], y_pred_sub[:, 2])
            print(f"| {sub_name} | {model_name} | {mae_cl:.4f} | {mae_cd:.4f} | {mae_cm:.4f} |")

def plot_polar_curves(test_df, xgb_models, mlp, cnn, solver):
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
    cst = np.concatenate([w_up, w_lo])
    
    alphas = np.arange(-5.0, 16.0, 1.0)
    Re = 1e6
    
    # Ground Truth Solver polars
    solver_results = [solver.solve(coords, a, Re) for a in alphas]
    cl_gt = [r['CL'] for r in solver_results]
    cd_gt = [r['CD'] for r in solver_results]
    cm_gt = [r['Cm'] for r in solver_results]
    
    # Construct input dataframe for XGBoost
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
    
    # XGBoost Predictions
    cl_xgb = xgb_models['CL'].predict(df_eval[feature_cols])
    cd_xgb = xgb_models['CD'].predict(df_eval[feature_cols])
    cm_xgb = xgb_models['Cm'].predict(df_eval[feature_cols])
    
    # PyTorch Predictions
    cst_t = torch.tensor(df_eval[[c for c in df_eval.columns if c.startswith('cst_')]].values, dtype=torch.float32)
    alpha_t = torch.tensor(df_eval['alpha'].values, dtype=torch.float32).unsqueeze(1)
    re_t = torch.tensor(df_eval['Re'].values, dtype=torch.float32).unsqueeze(1)
    
    mlp.cpu()
    cnn.cpu()
    with torch.no_grad():
        mlp_out = mlp(cst_t, alpha_t, re_t).numpy()
        
    coords_t = torch.tensor(coords.T, dtype=torch.float32).unsqueeze(0).repeat(len(alphas), 1, 1)
    with torch.no_grad():
        cnn_out = cnn(coords_t, alpha_t, re_t).numpy()
        
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # CL vs Alpha
    axes[0].plot(alphas, cl_gt, 'k-', linewidth=2, label='Solver (GT)')
    axes[0].plot(alphas, cl_xgb, 'r--', label='XGBoost')
    axes[0].plot(alphas, mlp_out[:, 0], 'g:', label='MLP (CST)')
    axes[0].plot(alphas, cnn_out[:, 0], 'b-.', label='CNN (Raw)')
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Lift Coefficient (CL)')
    axes[0].set_title('Lift Polar Curve')
    axes[0].grid(True)
    axes[0].legend()
    
    # CD vs Alpha
    axes[1].plot(alphas, cd_gt, 'k-', linewidth=2, label='Solver (GT)')
    axes[1].plot(alphas, cd_xgb, 'r--', label='XGBoost')
    axes[1].plot(alphas, mlp_out[:, 1], 'g:', label='MLP (CST)')
    axes[1].plot(alphas, cnn_out[:, 1], 'b-.', label='CNN (Raw)')
    axes[1].set_xlabel('Angle of Attack (deg)')
    axes[1].set_ylabel('Drag Coefficient (CD)')
    axes[1].set_title('Drag Polar Curve')
    axes[1].grid(True)
    axes[1].legend()
    
    # Cm vs Alpha
    axes[2].plot(alphas, cm_gt, 'k-', linewidth=2, label='Solver (GT)')
    axes[2].plot(alphas, cm_xgb, 'r--', label='XGBoost')
    axes[2].plot(alphas, mlp_out[:, 2], 'g:', label='MLP (CST)')
    axes[2].plot(alphas, cnn_out[:, 2], 'b-.', label='CNN (Raw)')
    axes[2].set_xlabel('Angle of Attack (deg)')
    axes[2].set_ylabel('Pitching Moment (Cm)')
    axes[2].set_title('Pitching Moment Polar Curve')
    axes[2].grid(True)
    axes[2].legend()
    
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
    _, _, test_df, _, _, _ = split_data_by_airfoil(df)
    
    # Find CST dimension from columns
    cst_dim = len([c for c in df.columns if c.startswith('cst_')])
    
    print("Loading models for evaluation...")
    lr, xgb_models, mlp, cnn = load_all_models(cst_dim=cst_dim)
    
    solver = AirfoilSolver()
    
    # Inference Speed Benchmark (using NACA 0012 coordinates from test_df if available)
    coords_dict = np.load("data/processed/airfoil_coords.npz")
    airfoil_name = test_df['airfoil_name'].iloc[0]
    coords = coords_dict[airfoil_name]
    evaluate_inference_speed(coords, 1e6, lr, xgb_models, mlp, cnn, solver)
    
    # Error analysis
    run_error_analysis(test_df, lr, xgb_models, mlp, cnn)
    
    # Plotting
    plot_polar_curves(test_df, xgb_models, mlp, cnn, solver)

if __name__ == "__main__":
    main()
