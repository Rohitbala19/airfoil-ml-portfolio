import numpy as np
from scipy.interpolate import interp1d

def parse_dat(content):
    """
    Parses UIUC coordinate format .dat files.
    Returns:
        name (str): Name of the airfoil
        coords (np.ndarray): Shape (N, 2) of x, y coordinates
    """
    lines = content.strip().split('\n')
    if not lines:
        return "Unknown", np.array([])
    
    # First line is usually the name
    name = lines[0].strip()
    
    coords = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                # Skip header lines that might contain numbers (e.g. number of points)
                if x > 10.0 or y > 10.0:
                    continue
                coords.append([x, y])
            except ValueError:
                # Line does not contain valid floats, skip it
                continue
                
    coords = np.array(coords)
    
    # If the first line was actually a coordinate, include it and reset name
    if len(lines[0].split()) >= 2:
        try:
            x = float(lines[0].split()[0])
            y = float(lines[0].split()[1])
            if x <= 10.0 and y <= 10.0:
                coords = np.vstack([[x, y], coords])
                name = "Unnamed Airfoil"
        except ValueError:
            pass
            
    return name, coords

def split_upper_lower(coords):
    """
    Splits airfoil coordinates into upper and lower surfaces starting from Leading Edge.
    Assumes coordinates start at Trailing Edge, go to Leading Edge, and return to Trailing Edge.
    """
    if len(coords) < 3:
        return coords, coords
        
    # Find Leading Edge (minimum x coordinate)
    le_idx = np.argmin(coords[:, 0])
    
    # Divide into two curves
    curve1 = coords[:le_idx + 1]
    curve2 = coords[le_idx:]
    
    # Identify which is upper and which is lower
    # We evaluate mean y-values of both curves (excluding LE and TE)
    # The curve with higher average y is the upper surface
    mean1 = np.mean(curve1[1:-1, 1]) if len(curve1) > 2 else np.mean(curve1[:, 1])
    mean2 = np.mean(curve2[1:-1, 1]) if len(curve2) > 2 else np.mean(curve2[:, 1])
    
    if mean1 > mean2:
        upper = curve1
        lower = curve2
    else:
        upper = curve2
        lower = curve1
        
    return upper, lower

def resample_airfoil(coords, num_points=100):
    """
    Resamples airfoil coordinates using cosine spacing.
    Returns:
        x_resampled (np.ndarray): Shape (2 * num_points,)
        y_resampled (np.ndarray): Shape (2 * num_points,)
    """
    if len(coords) < 3:
        # Return dummy arrays if coordinate set is empty/invalid
        x = np.linspace(1, 0, num_points)
        x = np.concatenate([x, x[::-1]])
        y = np.zeros_like(x)
        return x, y
        
    upper, lower = split_upper_lower(coords)
    
    # Normalize coordinates to chord length (c = 1.0, LE at x = 0.0)
    # Find LE and TE
    x_min = np.min(coords[:, 0])
    x_max = np.max(coords[:, 0])
    chord = x_max - x_min
    
    if chord <= 0:
        chord = 1.0
        
    upper_norm = (upper - [x_min, 0.0]) / [chord, 1.0]
    lower_norm = (lower - [x_min, 0.0]) / [chord, 1.0]
    
    # Cosine spacing grid
    beta = np.linspace(0, np.pi, num_points)
    x_grid = 0.5 * (1.0 - np.cos(beta)) # Cosine spaced from 0 to 1
    
    # Clean duplicates in coordinate curves for interpolation
    def clean_curve(curve):
        # Sort by x
        idx = np.argsort(curve[:, 0])
        x_sorted = curve[idx, 0]
        y_sorted = curve[idx, 1]
        
        # Ensure unique x for interpolation
        unique_idx = np.unique(x_sorted, return_index=True)[1]
        if len(unique_idx) < 2:
            return np.array([0.0, 1.0]), np.array([0.0, 0.0])
        return x_sorted[unique_idx], y_sorted[unique_idx]
    
    x_up, y_up = clean_curve(upper_norm)
    x_lo, y_lo = clean_curve(lower_norm)
    
    # Interpolators
    # Use fill_value="extrapolate" or bounds_error=False to handle endpoints safely
    f_up = interp1d(x_up, y_up, kind='linear', bounds_error=False, fill_value=(y_up[0], y_up[-1]))
    f_lo = interp1d(x_lo, y_lo, kind='linear', bounds_error=False, fill_value=(y_lo[0], y_lo[-1]))
    
    # Interpolate
    y_up_resampled = f_up(x_grid)
    y_lo_resampled = f_lo(x_grid)
    
    # Construct complete coordinate sequence (TE -> LE -> TE)
    # Upper goes TE (x=1) to LE (x=0)
    x_upper_seq = x_grid[::-1]
    y_upper_seq = y_up_resampled[::-1]
    
    # Lower goes LE (x=0) to TE (x=1)
    # We skip duplicate LE (x=0) to keep it a closed loop of unique points,
    # but for fixed length representations it's cleaner to have exactly N upper and N lower
    x_lower_seq = x_grid
    y_lower_seq = y_lo_resampled
    
    x_resampled = np.concatenate([x_upper_seq, x_lower_seq])
    y_resampled = np.concatenate([y_upper_seq, y_lower_seq])
    
    return x_resampled, y_resampled

def extract_geometric_features(coords):
    """
    Extracts key geometric features of the airfoil.
    Returns a dict of features:
        - max_thickness (t/c)
        - max_thickness_loc (x_t)
        - max_camber (m/c)
        - max_camber_loc (x_c)
        - le_radius (r_LE)
    """
    # Resample first to get a standardized representation
    x_res, y_res = resample_airfoil(coords, num_points=100)
    n = len(x_res) // 2
    
    # Upper goes from x=1 to x=0 (indices 0 to n-1, reversed)
    # Lower goes from x=0 to x=1 (indices n to 2n-1)
    x_grid = x_res[n:] # x from 0 to 1
    y_up = y_res[:n][::-1]
    y_lo = y_res[n:]
    
    # Camber and thickness
    thickness = y_up - y_lo
    camber = 0.5 * (y_up + y_lo)
    
    max_thickness = np.max(thickness)
    max_thickness_idx = np.argmax(thickness)
    max_thickness_loc = x_grid[max_thickness_idx]
    
    max_camber = np.max(camber)
    max_camber_idx = np.argmax(camber)
    max_camber_loc = x_grid[max_camber_idx]
    
    # Estimate Leading Edge Radius (r_LE)
    # Fit a parabola x = y^2 / (2 * r_LE) or circular fit for the nose region
    # We take a few points near the leading edge (e.g. index 0 to 5)
    # Since x is close to 0, let's use the upper surface nose points:
    x_nose = x_grid[1:6]
    y_nose = y_up[1:6]
    
    # x = a * y^2 => a = x / y^2
    # r_LE = 1 / (2 * a) = y^2 / (2 * x)
    if len(x_nose) > 0:
        r_estimates = (y_nose**2) / (2.0 * x_nose + 1e-8)
        le_radius = np.clip(np.mean(r_estimates), 0.001, 0.08)
    else:
        le_radius = 0.01
        
    return {
        'max_thickness': float(max_thickness),
        'max_thickness_loc': float(max_thickness_loc),
        'max_camber': float(max_camber),
        'max_camber_loc': float(max_camber_loc),
        'le_radius': float(le_radius)
    }

def fit_cst(coords, order=5):
    """
    Fits Class-Shape Transformation (CST) coefficients to the airfoil.
    Uses class function C(x) = x^0.5 * (1-x)^1.0.
    Returns:
        w_up (np.ndarray): Upper surface coefficients (shape: order + 1)
        w_lo (np.ndarray): Lower surface coefficients (shape: order + 1)
    """
    x_res, y_res = resample_airfoil(coords, num_points=100)
    n = len(x_res) // 2
    
    x_grid = x_res[n:]
    y_up = y_res[:n][::-1]
    y_lo = y_res[n:]
    
    # Bernstein polynomial basis functions
    def bernstein_basis(x, i, n_order):
        from scipy.special import comb
        return comb(n_order, i) * (x**i) * ((1.0 - x)**(n_order - i))
        
    # Class function
    C = np.sqrt(x_grid) * (1.0 - x_grid)
    
    # Avoid division by zero at endpoints
    active_idx = (x_grid > 0.0) & (x_grid < 1.0)
    x_act = x_grid[active_idx]
    y_up_act = y_up[active_idx]
    y_lo_act = y_lo[active_idx]
    C_act = C[active_idx]
    
    # Construct Design Matrix A
    # y(x) = C(x) * S(x) + x * y_TE  => S(x) = (y(x) - x*y_TE) / C(x)
    # S(x) = sum_i w_i * basis_i(x)
    A = np.zeros((len(x_act), order + 1))
    for i in range(order + 1):
        A[:, i] = bernstein_basis(x_act, i, order)
        
    # Subtract trailing edge thickness terms (assume y_TE is the endpoint value)
    y_TE_up = y_up[-1]
    y_TE_lo = y_lo[-1]
    
    target_up = (y_up_act - x_act * y_TE_up) / (C_act + 1e-8)
    target_lo = (y_lo_act - x_act * y_TE_lo) / (C_act + 1e-8)
    
    # Solve least squares
    w_up, _, _, _ = np.linalg.lstsq(A, target_up, rcond=None)
    w_lo, _, _, _ = np.linalg.lstsq(A, target_lo, rcond=None)
    
    return w_up, w_lo
