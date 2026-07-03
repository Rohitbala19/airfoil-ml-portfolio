import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import AirfoilMLP, AirfoilCNN1D, PolarSequencePredictor

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- PyTorch Dataset Definition ---
class AirfoilDataset(Dataset):
    """Dataset for Point-by-Point predictions (MLP and CNN)"""
    def __init__(self, df, coords_dict, device='cpu'):
        self.df = df.reset_index(drop=True)
        self.coords_dict = coords_dict
        self.device = device
        
        # Extracted geometry parameters
        self.geom_cols = ['max_thickness', 'max_thickness_loc', 'max_camber', 'max_camber_loc', 'le_radius']
        # CST parameters
        self.cst_cols = [c for c in df.columns if c.startswith('cst_')]
        
        # Features and Targets
        self.alphas = torch.tensor(self.df['alpha'].values, dtype=torch.float32).unsqueeze(1).to(device)
        self.res = torch.tensor(self.df['Re'].values, dtype=torch.float32).unsqueeze(1).to(device)
        self.cst = torch.tensor(self.df[self.cst_cols].values, dtype=torch.float32).to(device)
        
        self.targets = torch.tensor(self.df[['CL', 'CD', 'Cm']].values, dtype=torch.float32).to(device)
        
        # Pre-load coordinates to avoid CPU-GPU transfer overhead inside the loop
        self.names = self.df['airfoil_name'].values
        self.coords = []
        for name in self.names:
            coord = self.coords_dict[name] # shape (200, 2)
            # CNN Conv1d expects shape (in_channels=2, sequence_length=200)
            coord_t = torch.tensor(coord.T, dtype=torch.float32)
            self.coords.append(coord_t)
        self.coords = torch.stack(self.coords).to(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'coords': self.coords[idx],
            'cst': self.cst[idx],
            'alpha': self.alphas[idx],
            'Re': self.res[idx],
            'target': self.targets[idx]
        }

# --- Train/Val/Test Split by Airfoil Shape ---
def split_data_by_airfoil(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
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
    
    return train_df, val_df, test_df, train_names, val_names, test_names

# --- Train XGBoost Baselines ---
def train_xgboost(train_df, val_df, test_df):
    print("\n=== Training Tabular Baselines (Linear Regression & XGBoost) ===")
    
    # Feature engineering for tabular: alpha, Re, geometry parameters, CST parameters
    geom_cols = ['max_thickness', 'max_thickness_loc', 'max_camber', 'max_camber_loc', 'le_radius']
    cst_cols = [c for c in train_df.columns if c.startswith('cst_')]
    feature_cols = ['alpha', 'Re'] + geom_cols + cst_cols
    
    X_train = train_df[feature_cols]
    y_train = train_df[['CL', 'CD', 'Cm']]
    
    X_val = val_df[feature_cols]
    y_val = val_df[['CL', 'CD', 'Cm']]
    
    X_test = test_df[feature_cols]
    y_test = test_df[['CL', 'CD', 'Cm']]
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # XGBoost - train 3 separate models
    xgb_models = {}
    y_pred_xgb = np.zeros_like(y_test.values)
    
    os.makedirs("models/weights", exist_ok=True)
    
    for i, col in enumerate(['CL', 'CD', 'Cm']):
        print(f"Training XGBoost model for {col}...")
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(
            X_train, y_train[col],
            eval_set=[(X_val, y_val[col])],
            verbose=False
        )
        xgb_models[col] = model
        # Save model
        joblib.dump(model, f"models/weights/xgb_{col}.joblib")
        y_pred_xgb[:, i] = model.predict(X_test)
        
    joblib.dump(lr, "models/weights/linear_regression.joblib")
    
    # Print metrics
    print("\nTabular Baselines Evaluation on Unseen Airfoils:")
    for name, pred in [("Linear Regression", y_pred_lr), ("XGBoost", y_pred_xgb)]:
        print(f"  {name}:")
        for i, col in enumerate(['CL', 'CD', 'Cm']):
            mse = mean_squared_error(y_test.values[:, i], pred[:, i])
            mae = mean_absolute_error(y_test.values[:, i], pred[:, i])
            r2 = r2_score(y_test.values[:, i], pred[:, i])
            print(f"    {col} -> MSE: {mse:.6f}, MAE: {mae:.5f}, R2: {r2:.4f}")
            
    return xgb_models

# --- PyTorch Model Training ---
def train_pytorch_models(train_df, val_df, test_df, coords_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Training PyTorch Models on device: {device} ===")
    
    # 1. Create PyTorch datasets and loaders
    train_dataset = AirfoilDataset(train_df, coords_dict, device=device)
    val_dataset = AirfoilDataset(val_df, coords_dict, device=device)
    test_dataset = AirfoilDataset(test_df, coords_dict, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # 2. Instantiate Models
    cst_dim = len([c for c in train_df.columns if c.startswith('cst_')])
    mlp = AirfoilMLP(cst_dim=cst_dim).to(device)
    cnn = AirfoilCNN1D(coord_len=200).to(device)
    
    criterion = nn.MSELoss()
    
    # Train function for MLP and CNN
    def train_loop(model, name, epochs=100):
        print(f"\nTraining {name}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                if name == "AirfoilMLP":
                    pred = model(batch['cst'], batch['alpha'], batch['Re'])
                else: # CNN
                    pred = model(batch['coords'], batch['alpha'], batch['Re'])
                    
                loss = criterion(pred, batch['target'])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch['target'])
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    if name == "AirfoilMLP":
                        pred = model(batch['cst'], batch['alpha'], batch['Re'])
                    else: # CNN
                        pred = model(batch['coords'], batch['alpha'], batch['Re'])
                    loss = criterion(pred, batch['target'])
                    val_loss += loss.item() * len(batch['target'])
            val_loss /= len(val_loader.dataset)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"models/weights/{name.lower()}.pth")
                
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} (Best: {best_val_loss:.6f})")
                
        # Test evaluation
        model.load_state_dict(torch.load(f"models/weights/{name.lower()}.pth"))
        model.eval()
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for batch in test_loader:
                if name == "AirfoilMLP":
                    pred = model(batch['cst'], batch['alpha'], batch['Re'])
                else:
                    pred = model(batch['coords'], batch['alpha'], batch['Re'])
                test_preds.append(pred.cpu().numpy())
                test_targets.append(batch['target'].cpu().numpy())
                
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        
        print(f"{name} Evaluation on Unseen Airfoils:")
        for i, col in enumerate(['CL', 'CD', 'Cm']):
            mse = mean_squared_error(test_targets[:, i], test_preds[:, i])
            mae = mean_absolute_error(test_targets[:, i], test_preds[:, i])
            r2 = r2_score(test_targets[:, i], test_preds[:, i])
            print(f"    {col} -> MSE: {mse:.6f}, MAE: {mae:.5f}, R2: {r2:.4f}")
            
    train_loop(mlp, "AirfoilMLP", epochs=150)
    train_loop(cnn, "AirfoilCNN1D", epochs=150)
    
    # 3. Train PolarSequencePredictor
    # To predict the entire sequence of CL, CD, Cm for all 21 alphas at once:
    # We group training data by (airfoil_name, Re). Each sample yields a (21, 3) matrix
    train_seq_df = train_df.sort_values(['airfoil_name', 'Re', 'alpha'])
    val_seq_df = val_df.sort_values(['airfoil_name', 'Re', 'alpha'])
    test_seq_df = test_df.sort_values(['airfoil_name', 'Re', 'alpha'])
    
    # Ensure each combination has exactly 21 alphas
    # If not, let's interpolate or filter. Since we ran a uniform grid, they should match.
    def build_sequence_data(df, coords_dict):
        groups = df.groupby(['airfoil_name', 'Re'])
        cst_cols = [c for c in df.columns if c.startswith('cst_')]
        
        x_cst = []
        x_re = []
        y_seq = []
        
        for (name, Re), group in groups:
            if len(group) == 21: # Must match expected sequence size
                cst_vals = group[cst_cols].iloc[0].values
                targets = group[['CL', 'CD', 'Cm']].values # shape (21, 3)
                
                x_cst.append(cst_vals)
                x_re.append(Re)
                y_seq.append(targets)
                
        return (torch.tensor(np.array(x_cst), dtype=torch.float32).to(device),
                torch.tensor(np.array(x_re), dtype=torch.float32).unsqueeze(1).to(device),
                torch.tensor(np.array(y_seq), dtype=torch.float32).to(device))
                
    train_cst, train_re, train_y = build_sequence_data(train_seq_df, coords_dict)
    val_cst, val_re, val_y = build_sequence_data(val_seq_df, coords_dict)
    test_cst, test_re, test_y = build_sequence_data(test_seq_df, coords_dict)
    
    print(f"\nTraining PolarSequencePredictor with {len(train_cst)} sequence groups...")
    seq_predictor = PolarSequencePredictor(cst_dim=cst_dim).to(device)
    optimizer = torch.optim.AdamW(seq_predictor.parameters(), lr=0.003, weight_decay=1e-4)
    best_seq_val = float('inf')
    
    for epoch in range(120):
        seq_predictor.train()
        optimizer.zero_grad()
        pred = seq_predictor(train_cst, train_re)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        
        # Validation
        seq_predictor.eval()
        with torch.no_grad():
            val_pred = seq_predictor(val_cst, val_re)
            val_loss = criterion(val_pred, val_y)
            
        if val_loss.item() < best_seq_val:
            best_seq_val = val_loss.item()
            torch.save(seq_predictor.state_dict(), "models/weights/polarsequencepredictor.pth")
            
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:03d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f} (Best: {best_seq_val:.6f})")
            
    print("PolarSequencePredictor trained successfully.")

def main():
    dataset_path = "data/processed/airfoil_dataset.csv"
    coords_path = "data/processed/airfoil_coords.npz"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found. Run generator.py first!")
        return
        
    df = pd.read_csv(dataset_path)
    coords_dict = np.load(coords_path)
    
    # Split datasets
    train_df, val_df, test_df, _, _, _ = split_data_by_airfoil(df)
    
    # Train models
    train_xgboost(train_df, val_df, test_df)
    train_pytorch_models(train_df, val_df, test_df, coords_dict)
    
    print("\nTraining and validation runs completed successfully! All models saved to models/weights/")

if __name__ == "__main__":
    main()
