import os
import re
import sys
import ssl
import urllib.request
import numpy as np
import pandas as pd

# Workaround for macOS Python SSL certificate verification failures
ssl._create_default_https_context = ssl._create_unverified_context

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import parse_dat, resample_airfoil, extract_geometric_features, fit_cst
from src.solver import AirfoilSolver

def generate_naca_4_digit(m, p, t, num_points=100):
    """
    Generates coordinates for a NACA 4-digit airfoil analytical profile.
    m: max camber (camber ratio, e.g., 0.04)
    p: position of max camber (tenths of chord, e.g., 0.4)
    t: max thickness (thickness ratio, e.g., 0.12)
    """
    # Cosine spacing for high resolution at nose/tail
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1.0 - np.cos(beta))
    
    # Thickness distribution
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * (x**2) + 0.2843 * (x**3) - 0.1015 * (x**4))
    
    # Camber and camber slope
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if m > 0 and p > 0:
        # Front of max camber
        front_idx = x <= p
        yc[front_idx] = (m / (p**2)) * (2.0 * p * x[front_idx] - x[front_idx]**2)
        dyc_dx[front_idx] = (2.0 * m / (p**2)) * (p - x[front_idx])
        
        # Aft of max camber
        aft_idx = x > p
        yc[aft_idx] = (m / ((1.0 - p)**2)) * ((1.0 - 2.0 * p) + 2.0 * p * x[aft_idx] - x[aft_idx]**2)
        dyc_dx[aft_idx] = (2.0 * m / ((1.0 - p)**2)) * (p - x[aft_idx])
        
    theta = np.arctan(dyc_dx)
    
    x_up = x - yt * np.sin(theta)
    y_up = yc + yt * np.cos(theta)
    
    x_lo = x + yt * np.sin(theta)
    y_lo = yc - yt * np.cos(theta)
    
    # Combine to standard TE -> LE -> TE coordinates
    x_coords = np.concatenate([x_up[::-1], x_lo[1:]])
    y_coords = np.concatenate([y_up[::-1], y_lo[1:]])
    
    return np.column_stack([x_coords, y_coords])

def fetch_uiuc_airfoil_list():
    """Fetches the list of airfoils from the UIUC database site."""
    url = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            html = response.read().decode('utf-8', errors='ignore')
        # Find links to .dat files
        dat_files = re.findall(r'href="coord/([^"]+\.dat)"', html)
        return sorted(list(set(dat_files)))
    except Exception as e:
        print(f"Warning: Could not fetch UIUC airfoil list: {e}")
        return []

def download_uiuc_airfoil(filename, raw_dir):
    """Downloads a single airfoil coordinate file from UIUC database."""
    base_url = "https://m-selig.ae.illinois.edu/ads/coord/"
    url = base_url + filename
    dest_path = os.path.join(raw_dir, filename)
    
    # Avoid re-downloading if exists
    if os.path.exists(dest_path):
        return dest_path
        
    try:
        urllib.request.urlretrieve(url, dest_path)
        return dest_path
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

def build_airfoil_database(num_airfoils=40, raw_dir="data/raw", processed_path="data/processed/airfoil_dataset.csv"):
    """
    Downloads UIUC coordinates, generates diverse NACA coordinates as fallback/diversity,
    extracts features, fits CST, runs the aerodynamic solver, and exports the final dataset.
    """
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    airfoil_profiles = {}
    
    # 1. Try downloading some real UIUC airfoils
    print("Fetching UIUC airfoil coordinate database...")
    uiuc_list = fetch_uiuc_airfoil_list()
    
    # Choose a representative set of popular airfoils
    popular_uiuc = [
        "naca0012.dat", "naca4412.dat", "naca2412.dat", "clarky.dat", "s1223.dat", 
        "e387.dat", "dae11.dat", "fx60108.dat", "whitcomb.dat", "mh32.dat"
    ]
    
    # Add files from uiuc list up to limits
    download_list = popular_uiuc + [f for f in uiuc_list if f not in popular_uiuc]
    download_list = download_list[:num_airfoils]
    
    downloaded_count = 0
    for filename in download_list:
        path = download_uiuc_airfoil(filename, raw_dir)
        if path:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                name, coords = parse_dat(content)
                if len(coords) > 20: # Ensure valid parsing
                    airfoil_profiles[name] = coords
                    downloaded_count += 1
            except Exception as e:
                print(f"Error parsing downloaded airfoil {filename}: {e}")
                
    print(f"Successfully downloaded and parsed {downloaded_count} airfoils from UIUC.")
    
    # 2. Add synthetic NACA 4-digit airfoils to guarantee geometry coverage (e.g. 30 airfoils)
    # We span range of camber m: 0 to 6%, camber location p: 20% to 60%, thickness t: 6% to 18%
    naca_configs = []
    for m in [0.0, 0.02, 0.04, 0.06]:
        for p in [0.3, 0.4, 0.5] if m > 0 else [0.0]:
            for t in [0.08, 0.12, 0.16]:
                naca_configs.append((m, p, t))
                
    print(f"Generating {len(naca_configs)} analytical NACA 4-digit airfoils for dataset diversity...")
    for m, p, t in naca_configs:
        name = f"NACA {int(m*100)}{int(p*10)}{int(t*100):02d}"
        if name not in airfoil_profiles:
            coords = generate_naca_4_digit(m, p, t)
            airfoil_profiles[name] = coords
            
            # Save raw coordinate .dat for reference/reproducibility
            dat_path = os.path.join(raw_dir, f"{name.lower().replace(' ', '')}.dat")
            with open(dat_path, 'w') as f:
                f.write(f"{name}\n")
                for x, y in coords:
                    f.write(f"  {x:f}  {y:f}\n")
                    
    # 3. Process geometry features & CST fitting for all airfoils
    print("Performing geometry resampling, feature extraction, and CST fitting...")
    airfoil_metadata = {}
    for name, coords in airfoil_profiles.items():
        # Clean coordinates
        x_res, y_res = resample_airfoil(coords, num_points=100)
        geom = extract_geometric_features(coords)
        w_up, w_lo = fit_cst(coords, order=5)
        
        # Flatten CST coefficients for tabular storage
        cst_cols = {}
        for i, val in enumerate(w_up):
            cst_cols[f'cst_up_{i}'] = val
        for i, val in enumerate(w_lo):
            cst_cols[f'cst_lo_{i}'] = val
            
        airfoil_metadata[name] = {
            'coords': np.column_stack([x_res, y_res]),
            'features': geom,
            'cst': cst_cols
        }
        
    # 4. Aerodynamic simulation batch runner
    # We define parameters grid:
    # alphas: -5 to 15 degrees, step 1.0 (21 points)
    # Reynolds numbers: 1e5, 5e5, 1e6, 3e6
    alphas = np.arange(-5.0, 15.5, 1.0)
    reynolds = [1e5, 5e5, 1e6, 3e6]
    
    solver = AirfoilSolver()
    
    rows = []
    total_runs = len(airfoil_profiles) * len(alphas) * len(reynolds)
    print(f"Batch-running aerodynamic solver across {total_runs} conditions...")
    
    run_idx = 0
    failures = 0
    
    for name, meta in airfoil_metadata.items():
        coords = meta['coords']
        geom = meta['features']
        cst = meta['cst']
        
        for Re in reynolds:
            for alpha in alphas:
                run_idx += 1
                if run_idx % 500 == 0:
                    print(f"Progress: {run_idx}/{total_runs} sweeps completed...")
                    
                result = solver.solve(coords, alpha, Re)
                
                if not result['converged']:
                    failures += 1
                    # Log failure but continue (we keep the dummy or fallback representation)
                    
                row = {
                    'airfoil_name': name,
                    'alpha': alpha,
                    'Re': Re,
                    'CL': result['CL'],
                    'CD': result['CD'],
                    'Cm': result['Cm'],
                    'converged': int(result['converged']),
                    'solver_source': result['source'],
                    # Include geometric features
                    'max_thickness': geom['max_thickness'],
                    'max_thickness_loc': geom['max_thickness_loc'],
                    'max_camber': geom['max_camber'],
                    'max_camber_loc': geom['max_camber_loc'],
                    'le_radius': geom['le_radius'],
                }
                
                # Include CST coefficients
                row.update(cst)
                
                # Include raw coordinates as flattened string representation or save separate array
                # For tabular ML models we keep numeric columns, raw coordinates will be loaded
                # in PyTorch using the airfoil name maps.
                rows.append(row)
                
    df = pd.DataFrame(rows)
    df.to_csv(processed_path, index=False)
    print(f"Database generation complete. Total records: {len(df)}. Convergence failures: {failures}.")
    
    # Save the resampled coordinates mapping as a numpy npz file for deep learning models
    coords_dict = {name: meta['coords'] for name, meta in airfoil_metadata.items()}
    coords_save_path = os.path.join(os.path.dirname(processed_path), "airfoil_coords.npz")
    np.savez(coords_save_path, **coords_dict)
    print(f"Coordinates saved for deep learning training at {coords_save_path}.")
    
    return df

if __name__ == "__main__":
    build_airfoil_database(num_airfoils=20)
