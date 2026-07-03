# AeroML Surrogate Studio: Airfoil Aerodynamic ML Surrogate Model

A course portfolio-grade machine learning project implementing real-time aerodynamic coefficient prediction ($C_L, C_D, C_m$) of 2D airfoils as a function of geometry, angle of attack ($\alpha$), and Reynolds number ($Re$), benchmarked against physical solver data.

This project is tailored for **Fundamentals of Machine Intelligence and Data Science** courses, demonstrating standard machine learning regression algorithms, data preprocessing, feature engineering, group-based train/test splits, and model evaluations using standard metrics.

---

## 🛠️ Project Architecture

```
airfoil-ml-portfolio/
├── data/
│   ├── raw/                 # UIUC coordinate dat files
│   └── processed/           # Combined dataset (CSV) & resampled coordinates (NPZ)
├── src/
│   ├── geometry.py          # Coordinates resampling, CST fitting, geometric features
│   ├── solver.py            # Dual-mode solver interface (XFOIL + custom fallback)
│   ├── generator.py         # Batch runner to build database
│   ├── models.py            # Standard Scikit-Learn models definitions
│   ├── train.py             # Split-by-airfoil training loops
│   └── evaluate.py          # Inference speed benchmarks & error analysis
├── app/
│   └── main.py              # Streamlit interactive application
├── tests/
│   └── verify.py            # Validation script for geometric & solver functions
├── requirements.txt         # Project dependencies
└── README.md                # This documentation
```

---

## 📈 Core Data Science Concepts Covered

### 1. Data Cleaning & Feature Engineering (`src/geometry.py`)
- **Uniform Cosine Resampling:** Translates raw, unevenly spaced coordinates into exactly 200 points spaced using a cosine function, capturing high-curvature leading edges.
- **Dimensionality Reduction (CST Parametrization):** Instead of using 200 coordinate dimensions, we fit a 5th-order **Class-Shape Transformation (CST)** curve using **Linear Least Squares Regression** to represent the airfoil shape using just 12 Bernstein coefficients.
- **Hand-Crafted Features:** Extracts maximum thickness, camber, their chord locations, and leading-edge radius ($r_{LE}$) as explicit features.

### 2. Group-Based Validation (Avoiding Data Leakage)
In standard data science, splitting datasets randomly by row causes **data leakage** because parts of the same airfoil shape are shared between train and test sets. 
To ensure true evaluation of generalization, we split by **Airfoil Profile Group** (32 train, 7 validation, 7 test). The test set consists of completely *unseen* airfoil shapes, mimicking a real design setting.

### 3. Model Zoo: Fundamental Regressors (`src/models.py`)
We benchmark six fundamental supervised regression algorithms from **Scikit-Learn**:
1. **Linear Regression:** Baseline ordinary least squares model. Underfits significantly on non-linear components (like Drag).
2. **Polynomial Regression (Degree 2):** Maps features into a higher-dimensional polynomial space, capturing local quadratic interactions (e.g. lift-to-drag curves).
3. **K-Nearest Neighbors (KNN):** Distance-based non-parametric regressor.
4. **Decision Tree:** Single tree partition structure.
5. **Random Forest:** Ensemble bagging technique combining 100 decision trees to reduce variance.
6. **Multi-Layer Perceptron (MLP):** A fundamental fully connected Feedforward Artificial Neural Network trained using backpropagation.

---

## 🚀 Getting Started

### 1. Set Up Environment
```bash
# Clone the repository
git clone <your-repo-link>
cd airfoil-ml-portfolio

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate the Dataset
Compile the local dataset using our solver:
```bash
python3 src/generator.py
```

### 3. Run Verification Tests
```bash
python3 tests/verify.py
```

### 4. Train the Supervised Models
```bash
./run_training.sh
```
All models are standard standardized pipelines saved to `models/weights/`.

### 5. Run Benchmarks & Visualizations
```bash
./run_evaluation.sh
```
This performs inference speed comparisons and exports a comparison chart at `data/processed/polar_comparison.png`.

### 6. Launch the Interactive Studio
```bash
./run_studio.sh
```

---

## 📊 Performance Benchmarks (On Unseen Airfoils)

### Speedup Benchmark (Average Inference Time per Prediction)
| Model | Inference Time (ms) | Speedup Factor |
|---|---|---|
| Physics Solver | ~0.193 ms | 1.0x (Baseline) |
| Linear Regression | ~0.008 ms | ~23x Faster |
| Decision Tree | ~0.002 ms | ~73x Faster |
| MLP Regressor | ~0.004 ms | ~45x Faster |

### Error Analysis (MAE on Test Airfoils)
| Subset | Model | CL MAE | CD MAE | Cm MAE |
|---|---|---|---|---|
| **All Test Conditions** | Linear Regression | 0.0402 | 0.0107 | 0.0092 |
| | KNN Regressor | 0.0629 | 0.0020 | 0.0419 |
| | Random Forest | 0.0739 | 0.0017 | 0.0457 |
| | MLP Regressor | 0.0915 | 0.0254 | 0.0555 |

---

## 🎓 Key Learnings for Course Report
1. **Model Capacity vs. Generalization:** Simpler models (like KNN and Random Forest) often generalize better on small, structured physical datasets than deep neural networks (MLP) which can overfit training shapes easily.
2. **Feature Scaling Importance:** Distance-based models (KNN) and Gradient-based models (MLP) require standard normalization (`StandardScaler`), whereas Decision Trees are scale-invariant.
3. **Stall Regime Non-Linearity:** All models experience increased errors at high angles of attack ($\alpha \ge 10^\circ$), illustrating the physical limitations of surrogate regression during fluid separation.
