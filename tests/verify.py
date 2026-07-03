import sys
import os
import numpy as np

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import extract_geometric_features, fit_cst, resample_airfoil
from src.generator import generate_naca_4_digit
from src.solver import AirfoilSolver

def run_tests():
    print("=== Testing Geometry Modules ===")
    # Generate NACA 4412 (camber = 0.04, position = 0.4, thickness = 0.12)
    coords = generate_naca_4_digit(0.04, 0.4, 0.12, num_points=100)
    print(f"Generated NACA 4412 coordinates shape: {coords.shape}")
    
    # Extract features
    geom = extract_geometric_features(coords)
    print("Extracted Features:")
    for k, v in geom.items():
        print(f"  {k}: {v:.4f}")
        
    # Check values
    assert abs(geom['max_thickness'] - 0.12) < 0.01, f"Max thickness expected 0.12, got {geom['max_thickness']:.4f}"
    assert abs(geom['max_camber'] - 0.04) < 0.01, f"Max camber expected 0.04, got {geom['max_camber']:.4f}"
    print("Geometry feature extraction tests passed!")
    
    # Fit CST
    w_up, w_lo = fit_cst(coords, order=5)
    print(f"CST Coefficients (Upper): {np.round(w_up, 4)}")
    print(f"CST Coefficients (Lower): {np.round(w_lo, 4)}")
    assert len(w_up) == 6 and len(w_lo) == 6, "Expected 6 coefficients for order 5 CST"
    print("CST fitting tests passed!")
    
    print("\n=== Testing Aerodynamic Solver (Dual-Mode) ===")
    solver = AirfoilSolver()
    
    # Run a viscous solve for NACA 4412 at alpha = 5 degrees, Re = 1e6
    result = solver.solve(coords, alpha=5.0, Re=1e6)
    print("Aero Coefficients at alpha = 5.0, Re = 1e6:")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}")
        else:
            print(f"  {k}: {v}")
            
    # Check outputs for physically realistic ranges
    assert result['CL'] > 0.0, f"Lift coefficient should be positive for NACA 4412 at 5 deg, got {result['CL']:.5f}"
    assert result['CD'] > 0.0, f"Drag coefficient should be positive, got {result['CD']:.5f}"
    assert result['Cm'] < 0.0, f"Pitching moment should be negative for a cambered airfoil, got {result['Cm']:.5f}"
    print("Aerodynamic solver tests passed!")
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_tests()
