import torch
import torch.nn as nn
import torch.nn.functional as F

class AirfoilMLP(nn.Module):
    """
    Multilayer Perceptron mapping CST coefficients, alpha, and Reynolds number
    to aerodynamic coefficients (CL, CD, Cm).
    """
    def __init__(self, cst_dim=12, hidden_dim=128):
        super().__init__()
        # Input: 12 CST coefficients + alpha (1) + Re (1) = 14
        self.input_layer = nn.Linear(cst_dim + 2, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 3) # CL, CD, Cm
        
    def forward(self, cst, alpha, Re):
        # Scale alpha and Re for numerical stability
        # alpha is typically -5 to 15, Re is 1e5 to 5e6
        alpha_scaled = alpha / 15.0
        Re_scaled = torch.log10(Re) / 6.0
        
        # Concatenate inputs
        x = torch.cat([cst, alpha_scaled, Re_scaled], dim=-1)
        
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output_layer(x)

class AirfoilCNN1D(nn.Module):
    """
    1D Convolutional Neural Network operating on raw airfoil coordinates (shape: 2 x N),
    concatenated with alpha and Reynolds number, to predict aerodynamic coefficients.
    """
    def __init__(self, coord_len=200, hidden_dim=128):
        super().__init__()
        # Conv layers for shape processing
        # Input channel size = 2 (x, y coordinates)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Compute shape feature length after convolutions
        # 200 -> pool -> 100 -> pool -> 50 -> pool -> 25
        dummy_x = torch.zeros(1, 2, coord_len)
        with torch.no_grad():
            dummy_out = self.pool(F.relu(self.conv3(self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_x)))))))))
            flat_len = dummy_out.numel()
            
        self.fc_shape = nn.Linear(flat_len, hidden_dim)
        
        # Combine layers: shape features + alpha + Re
        self.fc_comb1 = nn.Linear(hidden_dim + 2, hidden_dim)
        self.fc_comb2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, 3) # CL, CD, Cm
        
    def forward(self, coords, alpha, Re):
        # coords shape: (batch_size, 2, coord_len)
        # alpha shape: (batch_size, 1)
        # Re shape: (batch_size, 1)
        
        # 1. Process coordinates with 1D Conv
        x_shape = self.pool(F.relu(self.conv1(coords)))
        x_shape = self.pool(F.relu(self.conv2(x_shape)))
        x_shape = self.pool(F.relu(self.conv3(x_shape)))
        
        x_shape = x_shape.view(x_shape.size(0), -1) # Flatten
        shape_feats = F.relu(self.fc_shape(x_shape))
        
        # 2. Scale features and concatenate
        alpha_scaled = alpha / 15.0
        Re_scaled = torch.log10(Re) / 6.0
        
        x_comb = torch.cat([shape_feats, alpha_scaled, Re_scaled], dim=-1)
        
        x = F.relu(self.fc_comb1(x_comb))
        x = F.relu(self.fc_comb2(x))
        return self.output_layer(x)

class PolarSequencePredictor(nn.Module):
    """
    Predicts the entire polar curve at once.
    Input: Geometry (CST coefficients) + Re
    Output: CL, CD, Cm across a pre-defined grid of angles of attack (21 values from -5 to 15 deg).
    """
    def __init__(self, cst_dim=12, num_alphas=21, hidden_dim=128):
        super().__init__()
        # Input: 12 CST + Re (1) = 13
        # Output: 21 alphas * 3 coefficients = 63 outputs
        self.input_layer = nn.Linear(cst_dim + 1, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_alphas * 3)
        self.num_alphas = num_alphas
        
    def forward(self, cst, Re):
        # Scale Re
        Re_scaled = torch.log10(Re) / 6.0
        
        x = torch.cat([cst, Re_scaled], dim=-1)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.output_layer(x)
        # Reshape to (batch_size, num_alphas, 3) where the 3 columns are CL, CD, Cm
        return out.view(-1, self.num_alphas, 3)
