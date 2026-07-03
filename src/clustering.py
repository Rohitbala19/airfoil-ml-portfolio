import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def perform_clustering(dataset_path="data/processed/airfoil_dataset.csv", n_clusters=3):
    """
    Applies K-Means and Gaussian Mixture Model (GMM) clustering to group airfoils
    unsupervisedly based on their geometric features.
    Uses PCA to reduce dimensions for 2D visualization.
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return None
        
    df = pd.read_csv(dataset_path)
    
    # Extract unique airfoils and their geometric features
    # Each airfoil should only have one row of geometry metadata
    geom_cols = ['max_thickness', 'max_thickness_loc', 'max_camber', 'max_camber_loc', 'le_radius']
    cst_cols = [c for c in df.columns if c.startswith('cst_')]
    feature_cols = geom_cols + cst_cols
    
    airfoil_df = df.groupby('airfoil_name')[feature_cols].first().reset_index()
    
    X = airfoil_df[feature_cols].values
    
    # 1. Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # 3. Gaussian Mixture Model (GMM) Clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    
    # Add labels back
    airfoil_df['kmeans_cluster'] = kmeans_labels
    airfoil_df['gmm_cluster'] = gmm_labels
    
    # 4. Dimensionality Reduction via PCA (for 2D plotting)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    airfoil_df['pca_1'] = X_pca[:, 0]
    airfoil_df['pca_2'] = X_pca[:, 1]
    
    # Plotting and saving the cluster visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot K-Means
    scatter0 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=60, edgecolors='k', alpha=0.8)
    axes[0].set_title(f"K-Means Clustering (K={n_clusters})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("PCA Component 1", color="#475569")
    axes[0].set_ylabel("PCA Component 2", color="#475569")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    # Add text labels for a few popular airfoils
    for idx, row in airfoil_df.iterrows():
        if row['airfoil_name'] in ['NACA 0012', 'NACA 4412', 'S1223', 'CLARK Y AIRFOIL', 'E387']:
            axes[0].annotate(row['airfoil_name'], (row['pca_1'], row['pca_2']), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
            
    # Plot GMM
    scatter1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='plasma', s=60, edgecolors='k', alpha=0.8)
    axes[1].set_title(f"Gaussian Mixture Model (GMM) (K={n_clusters})", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("PCA Component 1", color="#475569")
    axes[1].set_ylabel("PCA Component 2", color="#475569")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    # Add text labels for a few popular airfoils
    for idx, row in airfoil_df.iterrows():
        if row['airfoil_name'] in ['NACA 0012', 'NACA 4412', 'S1223', 'CLARK Y AIRFOIL', 'E387']:
            axes[1].annotate(row['airfoil_name'], (row['pca_1'], row['pca_2']), xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
            
    plt.suptitle("Unsupervised Clustering of Airfoil Geometries in 2D PCA Space", fontsize=14, y=0.98)
    plt.tight_layout()
    
    os.makedirs("data/processed", exist_ok=True)
    plot_path = "data/processed/geometry_clusters.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved clustering plot to {plot_path}")
    plt.close()
    
    # Save the clustering model parameters/labels for reference
    airfoil_df.to_csv("data/processed/airfoil_clusters.csv", index=False)
    print("Saved geometry clustering outputs to data/processed/airfoil_clusters.csv")
    
    return airfoil_df

if __name__ == "__main__":
    perform_clustering()
