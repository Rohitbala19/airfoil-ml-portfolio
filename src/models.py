from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def get_linear_model():
    """Ordinary Least Squares Linear Regression baseline."""
    return LinearRegression()

def get_polynomial_model(degree=2):
    """Polynomial Regression: mapping features to polynomial space, then applying Linear Regression."""
    return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())

def get_knn_model(n_neighbors=5):
    """K-Nearest Neighbors Regressor (distance-weighted)."""
    return KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')

def get_decision_tree(max_depth=10):
    """Decision Tree Regressor."""
    return DecisionTreeRegressor(max_depth=max_depth, random_state=42)

def get_random_forest(n_estimators=100, max_depth=12):
    """Random Forest Regressor (ensemble of decision trees)."""
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

def get_mlp_model(hidden_layer_sizes=(64, 64), max_iter=300):
    """
    Multi-Layer Perceptron (MLP) Feedforward Artificial Neural Network Regressor
    implemented in scikit-learn. Uses backpropagation (Adam solver) and ReLU activation.
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        activation='relu',
        solver='adam',
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    )
