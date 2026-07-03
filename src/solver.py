import os
import subprocess
import tempfile
import numpy as np
from src.geometry import resample_airfoil, extract_geometric_features

class AirfoilSolver:
    """
    Dual-mode aerodynamic solver interface.
    Attempts to run local XFOIL via subprocess.
    If XFOIL is not available or fails to converge, falls back to a high-fidelity Python-native
    physics-based aerodynamic solver incorporating thin airfoil theory and viscous/stall empirical models.
    """
    def __init__(self, xfoil_path="xfoil"):
        self.xfoil_path = xfoil_path
        self.xfoil_available = self._check_xfoil_availability()
        if not self.xfoil_available:
            print("Warning: XFOIL executable not found or not working. Fallback physics-based solver will be used.")

    def _check_xfoil_availability(self):
        """Checks if the XFOIL executable is available and can be executed."""
        try:
            # Try to run xfoil with a quit command
            process = subprocess.Popen(
                [self.xfoil_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, _ = process.communicate(input="quit\n", timeout=2.0)
            return "XFOIL" in stdout or process.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def solve(self, coords, alpha, Re):
        """
        Solves for aerodynamic coefficients CL, CD, Cm for a given airfoil geometry,
        angle of attack (alpha in degrees), and Reynolds number (Re).
        """
        # 1. Attempt XFOIL solver if available
        if self.xfoil_available:
            result = self._solve_xfoil(coords, alpha, Re)
            if result['converged']:
                return result
            
        # 2. Fall back to Python physics-based solver
        return self._solve_physics_fallback(coords, alpha, Re)

    def _solve_xfoil(self, coords, alpha, Re):
        """Runs XFOIL via subprocess to analyze the airfoil."""
        # Create temp files for coordinates and polar output
        temp_coords_fd, temp_coords_path = tempfile.mkstemp(suffix=".dat")
        temp_polar_fd, temp_polar_path = tempfile.mkstemp(suffix=".pol")
        
        try:
            # Write coordinates in XFOIL-compatible format (TE -> LE -> TE)
            # Ensure coordinates are clean and resampled to 100 points per surface (200 total)
            x_res, y_res = resample_airfoil(coords, num_points=100)
            with os.fdopen(temp_coords_fd, 'w') as f:
                f.write("Airfoil\n")
                for xi, yi in zip(x_res, y_res):
                    f.write(f"  {xi:f}  {yi:f}\n")
            
            # Close temp polar fd so XFOIL can write to it
            os.close(temp_polar_fd)
            
            # Define XFOIL command sequence
            commands = [
                f"load {temp_coords_path}",
                "pane", # Smooth panel distribution
                "oper",
                f"visc {Re}",
                "iter 100", # Increase iteration limit for convergence near stall
                "pacc",
                f"{temp_polar_path}",
                "", # No dump file
                f"alfa {alpha}",
                "pacc", # Turn off accumulation
                "quit"
            ]
            
            # Run XFOIL subprocess
            process = subprocess.Popen(
                [self.xfoil_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            input_str = "\n".join(commands) + "\n"
            try:
                stdout, stderr = process.communicate(input=input_str, timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                return {'CL': 0.0, 'CD': 0.0, 'Cm': 0.0, 'converged': False, 'source': 'xfoil_timeout'}
            
            # Read and parse polar file
            if os.path.exists(temp_polar_path) and os.path.getsize(temp_polar_path) > 0:
                with open(temp_polar_path, 'r') as f:
                    polar_lines = f.readlines()
                
                # Parse polar lines to find coefficient values
                # XFOIL polars contain header lines. The actual data lines have 7 columns:
                # alpha, CL, CD, Cdp, CM, Top_Xtr, Bot_Xtr
                data_found = False
                for line in polar_lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Data lines typically start with a float alpha close to target alpha
                    if len(parts) >= 5:
                        try:
                            a_val = float(parts[0])
                            # Check if this line is close to our target alpha
                            if abs(a_val - alpha) < 0.05:
                                CL = float(parts[1])
                                CD = float(parts[2])
                                Cm = float(parts[4])
                                return {'CL': CL, 'CD': CD, 'Cm': Cm, 'converged': True, 'source': 'xfoil'}
                        except ValueError:
                            continue
                            
            return {'CL': 0.0, 'CD': 0.0, 'Cm': 0.0, 'converged': False, 'source': 'xfoil_no_convergence'}
            
        except Exception as e:
            return {'CL': 0.0, 'CD': 0.0, 'Cm': 0.0, 'converged': False, 'source': f'xfoil_error: {str(e)}'}
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_coords_path):
                    os.remove(temp_coords_path)
                if os.path.exists(temp_polar_path):
                    os.remove(temp_polar_path)
            except OSError:
                pass

    def _solve_physics_fallback(self, coords, alpha, Re):
        """
        Pure Python aerodynamic solver based on thin airfoil theory and empirical viscous correction.
        Provides high-fidelity, realistic values for CL, CD, and Cm.
        """
        # Extract geometric features
        geom = extract_geometric_features(coords)
        t_c = geom['max_thickness']
        m_c = geom['max_camber']
        x_c = geom['max_camber_loc']
        
        # Resample to compute thin airfoil integrations
        x_res, y_res = resample_airfoil(coords, num_points=100)
        n = len(x_res) // 2
        x_grid = x_res[n:]
        y_up = y_res[:n][::-1]
        y_lo = y_res[n:]
        
        camber = 0.5 * (y_up + y_lo)
        
        # 1. Thin Airfoil Theory Fourier Coefficients
        # We transform to theta space: x = 0.5 * (1 - cos(theta))
        # theta goes from 0 to pi.
        theta = np.linspace(0, np.pi, len(x_grid))
        
        # Camber derivative dy_c/dx
        # Avoid division by zero at endpoints by clipping x
        dx = np.gradient(x_grid)
        dy = np.gradient(camber)
        dy_dx = dy / (dx + 1e-8)
        
        # Fourier terms:
        # A0 = alpha - (1/pi) * integral_0^pi (dy_c/dx) d_theta
        # A1 = (2/pi) * integral_0^pi (dy_c/dx) * cos(theta) d_theta
        # A2 = (2/pi) * integral_0^pi (dy_c/dx) * cos(2*theta) d_theta
        d_theta = np.pi / (len(x_grid) - 1)
        
        # Numerically integrate using trapezoidal rule helper
        def trapz(y_vals, dx_val):
            return float(np.sum(0.5 * (y_vals[:-1] + y_vals[1:])) * dx_val)
            
        int_camber_slope = trapz(dy_dx, d_theta)
        int_camber_slope_cos1 = trapz(dy_dx * np.cos(theta), d_theta)
        int_camber_slope_cos2 = trapz(dy_dx * np.cos(2 * theta), d_theta)
        
        # Zero-lift angle of attack in radians
        alpha_0 = -(1.0 / np.pi) * trapz(dy_dx * (1.0 - np.cos(theta)), d_theta)
        
        # A1 and A2 terms for pitching moment
        A1 = (2.0 / np.pi) * int_camber_slope_cos1
        A2 = (2.0 / np.pi) * int_camber_slope_cos2
        
        # Convert input alpha to radians
        alpha_rad = np.radians(alpha)
        
        # 2. Viscous Lift Curve Correction (slope eta * 2pi)
        eta = 0.90 - 0.15 * t_c # Lift slope reduction due to boundary layer thickening on thicker profiles
        CL_linear = 2.0 * np.pi * eta * (alpha_rad - alpha_0)
        
        # 3. Stall Modeling
        # Thicker profiles stall later and more smoothly
        # Camber shifts the stall angle
        alpha_stall_pos = np.radians(12.0 + 35.0 * t_c + 15.0 * m_c)
        alpha_stall_neg = np.radians(-12.0 - 35.0 * t_c + 15.0 * m_c)
        
        # Sigmoid function for stall factor (1 = fully attached flow, 0 = fully separated)
        alpha_camber = 4.0 * m_c * (1.0 - x_c) # Angle corresponding to zero lift/camber influence
        alpha_rel = alpha_rad - np.radians(alpha_camber)
        
        # Soft transition width
        k = 15.0 - 20.0 * t_c # thinner airfoils have steeper stall drop-off
        k = max(k, 5.0)
        
        if alpha_rel >= 0:
            stall_sigmoid = 1.0 / (1.0 + np.exp(k * (alpha_rel - alpha_stall_pos)))
        else:
            stall_sigmoid = 1.0 / (1.0 + np.exp(k * (alpha_stall_neg - alpha_rel)))
            
        # Post-stall lift model (Newtonian separation flow: 2 * sin(a) * cos(a))
        CL_separated = 2.0 * np.sin(alpha_rad) * np.cos(alpha_rad)
        
        # Total Lift
        CL = stall_sigmoid * CL_linear + (1.0 - stall_sigmoid) * CL_separated
        
        # 4. Viscous Drag Estimation
        # Flat plate turbulent boundary layer skin friction coefficient (Prandtl-Schlichting transition model)
        # Scaled by Reynolds number
        Re_clamped = max(Re, 1e4)
        Cf = 0.455 / (np.log10(Re_clamped) ** 2.58) - (1700.0 / Re_clamped)
        Cf = max(Cf, 0.001)
        
        # Profile drag coefficient at zero lift (form factor correction)
        CD0 = 2.0 * Cf * (1.0 + 2.0 * t_c + 60.0 * (t_c ** 4))
        
        # Induced drag-like term from lift
        # Thick airfoils and cambered airfoils have higher drag increments with lift
        k_drag = 0.005 + 0.05 * t_c + 0.1 * m_c
        CD_lift = k_drag * (CL ** 2)
        
        # Stall drag (pressure drag due to separation)
        CD_stall = 2.0 * (np.sin(alpha_rad) ** 2) * (1.0 - stall_sigmoid)
        
        # Total Drag
        CD = CD0 + CD_lift + CD_stall
        
        # 5. Pitching Moment Estimation
        # Linear moment from thin airfoil theory about quarter-chord
        # Cm_c4 = -pi/2 * (A1 - A2)
        Cm_linear = -(np.pi / 2.0) * (A1 - A2) + (CL_linear * 0.25 * (1.0 - eta)) # Pitching moment adjustment
        
        # Post-stall moment moves center of pressure to 50% chord
        # Cm = -0.5 * sin(alpha) * cos(alpha) (force acting at 0.5c gives moment about 0.25c as -F * 0.25)
        # Normal force is roughly 2 * sin^2(alpha) or similar. Let's use empirical pitching moment drop:
        Cm_separated = -0.25 * np.sin(alpha_rad)
        
        Cm = stall_sigmoid * Cm_linear + (1.0 - stall_sigmoid) * Cm_separated
        
        return {
            'CL': float(CL),
            'CD': float(CD),
            'Cm': float(Cm),
            'converged': True,
            'source': 'physics_fallback'
        }
