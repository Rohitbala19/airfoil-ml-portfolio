import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import extract_geometric_features, fit_cst, resample_airfoil
from src.generator import generate_naca_4_digit
from src.solver import AirfoilSolver
from src.models import AirfoilMLP, AirfoilCNN1D

# Set Page Config
st.set_page_config(
    page_title="AeroML - Aerodynamic Surrogate Studio",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Neon Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Core Layout Styles */
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Title Header */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .title-main {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #6366f1, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        text-shadow: 0 0 40px rgba(99, 102, 241, 0.3);
    }
    
    .subtitle-main {
        font-size: 1.15rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Sidebar Design */
    [data-testid="stSidebar"] {
        background-color: #12151c !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Glassmorphic Metric Cards */
    .metric-card {
        background: rgba(22, 28, 41, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15);
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-source {
        font-size: 0.75rem;
        color: #38bdf8;
        font-weight: 500;
    }
    
    /* Speedup glowing badge */
    .speedup-badge {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.1);
    }
    
    .speedup-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #c084fc;
    }
    
    /* Style inputs */
    div[data-baseweb="select"] > div {
        background-color: #1a1e27 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    
    /* Subheadings */
    h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load all models
@st.cache_resource
def load_surrogate_models():
    models_dir = "models/weights"
    if not os.path.exists(models_dir):
        return None, None, None
        
    try:
        # Load XGBoost models
        xgb_models = {}
        for col in ['CL', 'CD', 'Cm']:
            path = os.path.join(models_dir, f"xgb_{col}.joblib")
            if os.path.exists(path):
                xgb_models[col] = joblib.load(path)
            else:
                return None, None, None
                
        # Load MLP
        mlp = AirfoilMLP(cst_dim=12)
        mlp_path = os.path.join(models_dir, "airfoilmlp.pth")
        if os.path.exists(mlp_path):
            mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
            mlp.eval()
        else:
            mlp = None
            
        # Load CNN
        cnn = AirfoilCNN1D(coord_len=200)
        cnn_path = os.path.join(models_dir, "airfoilcnn1d.pth")
        if os.path.exists(cnn_path):
            cnn.load_state_dict(torch.load(cnn_path, map_location='cpu'))
            cnn.eval()
        else:
            cnn = None
            
        return xgb_models, mlp, cnn
    except Exception as e:
        st.warning(f"Could not load trained models: {e}. Running solver mode only.")
        return None, None, None

def main():
    # Header Title
    st.markdown("""
    <div class="header-container">
        <h1 class="title-main">AeroML Surrogate Studio</h1>
        <p class="subtitle-main">Deep learning & machine learning aerodynamic surrogate modeling in real-time, benchmarked against XFOIL physics</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Sidebar Panel
    st.sidebar.markdown("### 🛠️ Config & Airfoil Design")
    
    airfoil_mode = st.sidebar.radio(
        "Airfoil Selection Mode",
        ["NACA 4-Digit Designer", "Database Presets", "Upload Custom Coordinate File (.dat)"]
    )
    
    coords = None
    airfoil_name = ""
    
    if airfoil_mode == "NACA 4-Digit Designer":
        # Interactive sliders to design NACA profiles
        m_slider = st.sidebar.slider("Max Camber (m)", 0.0, 0.08, 0.04, 0.01, format="%.2f")
        p_slider = st.sidebar.slider("Max Camber Position (p)", 0.2, 0.8, 0.4, 0.1, format="%.1f")
        t_slider = st.sidebar.slider("Max Thickness (t)", 0.05, 0.25, 0.12, 0.01, format="%.2f")
        
        airfoil_name = f"NACA {int(m_slider*100)}{int(p_slider*10)}{int(t_slider*100):02d}"
        coords = generate_naca_4_digit(m_slider, p_slider, t_slider, num_points=100)
        
    elif airfoil_mode == "Database Presets":
        presets = {
            "NACA 0012 (Symmetric)": "naca0012",
            "NACA 4412 (Standard Wing)": "naca4412",
            "NACA 2412 (General Aviation)": "naca2412",
            "Selig S1223 (High Lift / UAV)": "s1223",
            "Clark Y (Vintage / Propeller)": "clarky",
            "Eppler E387 (Low Re Windtunnel)": "e387",
            "DAE11 (Daedalus Human Powered)": "dae11",
            "Whitcomb IL71 (Supercritical)": "whitcomb"
        }
        
        selected_preset = st.sidebar.selectbox("Choose Airfoil Preset", list(presets.keys()))
        name_key = presets[selected_preset]
        
        # Load preset from data/raw or generate analytically if raw missing
        dat_path = f"data/raw/{name_key}.dat"
        if os.path.exists(dat_path):
            with open(dat_path, 'r') as f:
                content = f.read()
            from src.geometry import parse_dat
            airfoil_name, coords = parse_dat(content)
        else:
            # Fall back to analytic generation for NACA presets if dat missing
            if name_key == "naca0012":
                coords = generate_naca_4_digit(0.0, 0.0, 0.12)
            elif name_key == "naca4412":
                coords = generate_naca_4_digit(0.04, 0.4, 0.12)
            elif name_key == "naca2412":
                coords = generate_naca_4_digit(0.02, 0.4, 0.12)
            else:
                st.sidebar.error("Preset file missing in data/raw. Falling back to NACA 0012.")
                coords = generate_naca_4_digit(0.0, 0.0, 0.12)
            airfoil_name = selected_preset
            
    else: # File Upload
        uploaded_file = st.sidebar.file_uploader("Upload Coordinate File (.dat)", type=["dat", "txt"])
        if uploaded_file is not None:
            content = uploaded_file.read().decode("utf-8")
            from src.geometry import parse_dat
            airfoil_name, coords = parse_dat(content)
        else:
            st.sidebar.info("Upload an airfoil file to start. Using NACA 0012 baseline.")
            airfoil_name = "NACA 0012"
            coords = generate_naca_4_digit(0.0, 0.0, 0.12)
            
    # Geometry Analysis
    geom = extract_geometric_features(coords)
    w_up, w_lo = fit_cst(coords, order=5)
    cst = np.concatenate([w_up, w_lo])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌬️ Flow Conditions")
    alpha = st.sidebar.slider("Angle of Attack (α)", -6.0, 16.0, 5.0, 0.5, format="%.1f°")
    Re = st.sidebar.select_slider(
        "Reynolds Number (Re)",
        options=[100000, 500000, 1000000, 3000000],
        value=1000000,
        format_func=lambda x: f"{x:,}"
    )
    
    # 2. Main Page Columns
    col_geom, col_metrics = st.columns([1.1, 0.9])
    
    with col_geom:
        st.markdown(f"### 📐 Geometry: {airfoil_name}")
        
        # Resample for graphing
        x_res, y_res = resample_airfoil(coords, num_points=100)
        
        fig_shape = go.Figure()
        # Plot airfoil surface
        fig_shape.add_trace(go.Scatter(
            x=x_res, y=y_res,
            mode='lines',
            line=dict(color='#38bdf8', width=3.5),
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.05)',
            name='Airfoil Surface'
        ))
        # Plot camber line
        n = len(x_res) // 2
        x_grid = x_res[n:]
        y_up = y_res[:n][::-1]
        y_lo = y_res[n:]
        y_camber = 0.5 * (y_up + y_lo)
        
        fig_shape.add_trace(go.Scatter(
            x=x_grid, y=y_camber,
            mode='lines',
            line=dict(color='#a855f7', width=2, dash='dash'),
            name='Camber Line'
        ))
        
        fig_shape.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', scaleanchor="y", scaleratio=1, title="Chord Location (x/c)", color="#94a3b8"),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Thickness (y/c)", color="#94a3b8"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=280,
            showlegend=True,
            legend=dict(font=dict(color='#94a3b8'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_shape, use_container_width=True)
        
        # Display extracted features in columns
        f_col1, f_col2, f_col3 = st.columns(3)
        f_col1.metric("Max Thickness (t/c)", f"{geom['max_thickness']*100:.2f}%", f"at {geom['max_thickness_loc']*100:.1f}% c")
        f_col2.metric("Max Camber (m/c)", f"{geom['max_camber']*100:.2f}%", f"at {geom['max_camber_loc']*100:.1f}% c")
        f_col3.metric("LE Radius (r_LE)", f"{geom['le_radius']*100:.3f}% c")
        
    # Aerodynamic Computations
    solver = AirfoilSolver()
    
    # 1. Physics solver benchmark (ground truth)
    t_start = time.perf_counter()
    phys_result = solver.solve(coords, alpha, Re)
    t_phys = time.perf_counter() - t_start
    
    # Try loading ML model predictions
    xgb_models, mlp, cnn = load_surrogate_models()
    
    # Prepare input arrays for ML
    xgb_val = None
    mlp_val = None
    cnn_val = None
    
    t_ml = 0
    
    if xgb_models is not None:
        # Create input features
        feature_dict = {
            'alpha': alpha,
            'Re': Re,
            'max_thickness': geom['max_thickness'],
            'max_thickness_loc': geom['max_thickness_loc'],
            'max_camber': geom['max_camber'],
            'max_camber_loc': geom['max_camber_loc'],
            'le_radius': geom['le_radius']
        }
        for i, v in enumerate(w_up):
            feature_dict[f'cst_up_{i}'] = v
        for i, v in enumerate(w_lo):
            feature_dict[f'cst_lo_{i}'] = v
            
        df_ml = pd.DataFrame([feature_dict])
        feature_cols = ['alpha', 'Re'] + list(geom.keys()) + [f'cst_up_{i}' for i in range(6)] + [f'cst_lo_{i}' for i in range(6)]
        X_ml = df_ml[feature_cols]
        
        # Measure ML speed (XGBoost, MLP, CNN cumulative or individual)
        t_ml_start = time.perf_counter()
        
        # Run predictions
        cl_xgb = xgb_models['CL'].predict(X_ml)[0]
        cd_xgb = xgb_models['CD'].predict(X_ml)[0]
        cm_xgb = xgb_models['Cm'].predict(X_ml)[0]
        
        if mlp is not None:
            cst_t = torch.tensor(X_ml[[c for c in X_ml.columns if c.startswith('cst_')]].values, dtype=torch.float32)
            alpha_t = torch.tensor([alpha], dtype=torch.float32).unsqueeze(1)
            re_t = torch.tensor([Re], dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                mlp_out = mlp(cst_t, alpha_t, re_t).numpy()[0]
        else:
            mlp_out = None
            
        if cnn is not None:
            coords_t = torch.tensor(coords.T, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                cnn_out = cnn(coords_t, alpha_t, re_t).numpy()[0]
        else:
            cnn_out = None
            
        t_ml = time.perf_counter() - t_ml_start
        
    with col_metrics:
        st.markdown("### ⚡ Live Aero Coefficients")
        
        # Display in columns
        c_l, c_d, c_m = st.columns(3)
        
        # We display the selected model predictions or the physics solver
        model_choice = st.selectbox(
            "Select Prediction Engine",
            ["XFOIL / Physics Solver", "XGBoost Surrogate", "PyTorch MLP (CST)", "PyTorch CNN (Raw Coords)"] if xgb_models is not None else ["XFOIL / Physics Solver"]
        )
        
        val_cl, val_cd, val_cm = 0.0, 0.0, 0.0
        engine_source = ""
        
        if model_choice == "XFOIL / Physics Solver":
            val_cl, val_cd, val_cm = phys_result['CL'], phys_result['CD'], phys_result['Cm']
            engine_source = f"Physics Model ({phys_result['source']})"
            infer_time = t_phys
        elif model_choice == "XGBoost Surrogate":
            val_cl, val_cd, val_cm = cl_xgb, cd_xgb, cm_xgb
            engine_source = "XGBoost Tabular"
            infer_time = t_ml / 3.0
        elif model_choice == "PyTorch MLP (CST)":
            val_cl, val_cd, val_cm = mlp_out[0], mlp_out[1], mlp_out[2]
            engine_source = "Deep Learning MLP"
            infer_time = t_ml / 3.0
        else:
            val_cl, val_cd, val_cm = cnn_out[0], cnn_out[1], cnn_out[2]
            engine_source = "Deep Learning 1D CNN"
            infer_time = t_ml / 3.0
            
        c_l.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Lift Coefficient (C<sub>L</sub>)</div>
            <div class="metric-value">{val_cl:.4f}</div>
            <div class="metric-source">{engine_source}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c_d.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Drag Coefficient (C<sub>D</sub>)</div>
            <div class="metric-value">{val_cd:.5f}</div>
            <div class="metric-source">{engine_source}</div>
        </div>
        """, unsafe_allow_html=True)
        
        c_m.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Moment Coefficient (C<sub>m</sub>)</div>
            <div class="metric-value">{val_cm:.4f}</div>
            <div class="metric-source">{engine_source}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display inference speed and speedup
        st.markdown("---")
        speed_col1, speed_col2 = st.columns(2)
        
        speed_col1.markdown(f"""
        <div style="background:rgba(22,28,41,0.4); padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,0.05); text-align:center;">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Inference Time</div>
            <div style="font-size:1.35rem; font-weight:700; color:white; margin-top:0.25rem;">{infer_time * 1000:.3f} ms</div>
        </div>
        """, unsafe_allow_html=True)
        
        if model_choice != "XFOIL / Physics Solver" and t_ml > 0:
            speedup = t_phys / (infer_time + 1e-9)
            speed_col2.markdown(f"""
            <div class="speedup-badge">
                <div style="font-size:0.75rem; color:#c084fc; text-transform:uppercase; font-weight:600;">Surrogate Speedup</div>
                <div class="speedup-value">{speedup:,.1f}x Faster</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            speed_col2.markdown(f"""
            <div style="background:rgba(22,28,41,0.4); padding:1rem; border-radius:12px; border:1px solid rgba(255,255,255,0.05); text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase; font-weight:600;">Physics Solver Time</div>
                <div style="font-size:1.35rem; font-weight:700; color:#38bdf8; margin-top:0.25rem;">{t_phys * 1000:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
            
    # 3. Polar curves row
    st.markdown("### 📊 Aerodynamic Polar Performance Curves")
    
    # Calculate polars across alpha range
    alphas_polar = np.linspace(-6.0, 16.0, 23)
    
    # Physics Solver (Ground Truth) curves
    cl_gt_polar = []
    cd_gt_polar = []
    cm_gt_polar = []
    
    for a in alphas_polar:
        res = solver.solve(coords, a, Re)
        cl_gt_polar.append(res['CL'])
        cd_gt_polar.append(res['CD'])
        cm_gt_polar.append(res['Cm'])
        
    # Generate ML polars if models are available
    show_ml_polars = (xgb_models is not None)
    
    if show_ml_polars:
        rows_polar = []
        for a in alphas_polar:
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
            rows_polar.append(row)
            
        df_polar = pd.DataFrame(rows_polar)
        
        # XGBoost
        cl_xgb_polar = xgb_models['CL'].predict(df_polar[feature_cols])
        cd_xgb_polar = xgb_models['CD'].predict(df_polar[feature_cols])
        cm_xgb_polar = xgb_models['Cm'].predict(df_polar[feature_cols])
        
        # PyTorch MLP
        if mlp is not None:
            cst_t_polar = torch.tensor(df_polar[[c for c in df_polar.columns if c.startswith('cst_')]].values, dtype=torch.float32)
            alpha_t_polar = torch.tensor(df_polar['alpha'].values, dtype=torch.float32).unsqueeze(1)
            re_t_polar = torch.tensor(df_polar['Re'].values, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                mlp_out_polar = mlp(cst_t_polar, alpha_t_polar, re_t_polar).numpy()
        
        # PyTorch CNN
        if cnn is not None:
            coords_t_polar = torch.tensor(coords.T, dtype=torch.float32).unsqueeze(0).repeat(len(alphas_polar), 1, 1)
            with torch.no_grad():
                cnn_out_polar = cnn(coords_t_polar, alpha_t_polar, re_t_polar).numpy()
                
    # Plotly figures layout (3 subplots in a row)
    fig_polars = make_subplots(rows=1, cols=3, subplot_titles=(
        "Lift Polar (C<sub>L</sub> vs &alpha;)",
        "Drag Polar (C<sub>D</sub> vs &alpha;)",
        "Drag Polar Envelope (C<sub>L</sub> vs C<sub>D</sub>)"
    ))
    
    # Color scheme
    colors = {
        'gt': '#f1f5f9',
        'xgb': '#ef4444',
        'mlp': '#22c55e',
        'cnn': '#a855f7'
    }
    
    # Trace 1: CL vs Alpha
    fig_polars.add_trace(go.Scatter(x=alphas_polar, y=cl_gt_polar, name="Physics Solver (GT)", line=dict(color=colors['gt'], width=2.5)), row=1, col=1)
    if show_ml_polars:
        fig_polars.add_trace(go.Scatter(x=alphas_polar, y=cl_xgb_polar, name="XGBoost", line=dict(color=colors['xgb'], dash='dash', width=2)), row=1, col=1)
        if mlp is not None:
            fig_polars.add_trace(go.Scatter(x=alphas_polar, y=mlp_out_polar[:, 0], name="PyTorch MLP", line=dict(color=colors['mlp'], dash='dot', width=2)), row=1, col=1)
        if cnn is not None:
            fig_polars.add_trace(go.Scatter(x=alphas_polar, y=cnn_out_polar[:, 0], name="PyTorch CNN", line=dict(color=colors['cnn'], dash='dashdot', width=2)), row=1, col=1)
            
    # Trace 2: CD vs Alpha
    fig_polars.add_trace(go.Scatter(x=alphas_polar, y=cd_gt_polar, name="Physics Solver (GT)", line=dict(color=colors['gt'], width=2.5), showlegend=False), row=1, col=2)
    if show_ml_polars:
        fig_polars.add_trace(go.Scatter(x=alphas_polar, y=cd_xgb_polar, name="XGBoost", line=dict(color=colors['xgb'], dash='dash', width=2), showlegend=False), row=1, col=2)
        if mlp is not None:
            fig_polars.add_trace(go.Scatter(x=alphas_polar, y=mlp_out_polar[:, 1], name="PyTorch MLP", line=dict(color=colors['mlp'], dash='dot', width=2), showlegend=False), row=1, col=2)
        if cnn is not None:
            fig_polars.add_trace(go.Scatter(x=alphas_polar, y=cnn_out_polar[:, 1], name="PyTorch CNN", line=dict(color=colors['cnn'], dash='dashdot', width=2), showlegend=False), row=1, col=2)
            
    # Trace 3: CL vs CD (Drag Polar)
    fig_polars.add_trace(go.Scatter(x=cd_gt_polar, y=cl_gt_polar, name="Physics Solver (GT)", line=dict(color=colors['gt'], width=2.5), showlegend=False), row=1, col=3)
    if show_ml_polars:
        fig_polars.add_trace(go.Scatter(x=cd_xgb_polar, y=cl_xgb_polar, name="XGBoost", line=dict(color=colors['xgb'], dash='dash', width=2), showlegend=False), row=1, col=3)
        if mlp is not None:
            fig_polars.add_trace(go.Scatter(x=mlp_out_polar[:, 1], y=mlp_out_polar[:, 0], name="PyTorch MLP", line=dict(color=colors['mlp'], dash='dot', width=2), showlegend=False), row=1, col=3)
        if cnn is not None:
            fig_polars.add_trace(go.Scatter(x=cnn_out_polar[:, 1], y=cnn_out_polar[:, 0], name="PyTorch CNN", line=dict(color=colors['cnn'], dash='dashdot', width=2), showlegend=False), row=1, col=3)
            
    # Style plots
    fig_polars.update_layout(
        plot_bgcolor='rgba(15,23,42,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color='#94a3b8')
        ),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="AoA (deg)", color="#94a3b8"),
        xaxis2=dict(gridcolor='rgba(255,255,255,0.05)', title="AoA (deg)", color="#94a3b8"),
        xaxis3=dict(gridcolor='rgba(255,255,255,0.05)', title="Drag Coeff (CD)", color="#94a3b8"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Lift Coeff (CL)", color="#94a3b8"),
        yaxis2=dict(gridcolor='rgba(255,255,255,0.05)', title="Drag Coeff (CD)", color="#94a3b8"),
        yaxis3=dict(gridcolor='rgba(255,255,255,0.05)', title="Lift Coeff (CL)", color="#94a3b8")
    )
    
    st.plotly_chart(fig_polars, use_container_width=True)
    
    # 4. Error Analysis summary banner
    if not show_ml_polars:
        st.info("💡 Pro-Tip: Once you finish training the ML models, they will automatically be displayed here. To start training, run: `python3 src/train.py`.")
    else:
        st.success("🤖 Machine learning surrogate models are active! The MLP uses Class-Shape Transformation (CST) geometry representation, while the 1D CNN processes raw coordinate shape signals directly.")

if __name__ == "__main__":
    main()
