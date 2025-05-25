import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

def find_optimal_k(df, max_k=10):
    """Find optimal number of clusters using silhouette score."""
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    X = df[share_columns].values
    
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    return optimal_k

def perform_kmeans_clustering(df, n_clusters):
    """Perform K-means clustering and analyze results."""
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    X = df[share_columns].values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Analyze cluster characteristics
    cluster_analysis = {}
    for cluster in range(n_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        cluster_analysis[cluster] = {
            'size': len(cluster_data),
            'locations': cluster_data['Location'].tolist(),
            'mean_proportions': cluster_data[share_columns].mean(),
            'mean_diversity': cluster_data['Diversity_Index'].mean(),
            'mean_heavy_ratio': cluster_data['Heavy_Vehicle_Ratio'].mean()
        }
    
    return df, cluster_analysis

def perform_hierarchical_clustering(df, n_clusters):
    """Perform hierarchical clustering."""
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    X = df[share_columns].values
    
    # Perform clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    df['HC_Cluster'] = hc.fit_predict(X)
    
    return df

def detect_anomalies(df, method='isolation_forest'):
    """Detect anomalies using either Isolation Forest or Local Outlier Factor."""
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    X = df[share_columns].values
    
    if method == 'isolation_forest':
        model = IsolationForest(contamination=0.1, random_state=42)
        df['Anomaly_Score'] = -model.fit_predict(X)  # Convert to positive scores
    else:  # LOF
        model = LocalOutlierFactor(contamination=0.1)
        df['Anomaly_Score'] = -model.fit_predict(X)  # Convert to positive scores
    
    # Identify top anomalies
    anomalies = df.nlargest(3, 'Anomaly_Score')
    
    return df, anomalies

def plot_cluster_characteristics(cluster_analysis, save_path=None):
    """Plot characteristics of each cluster."""
    n_clusters = len(cluster_analysis)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data for plotting
    cluster_sizes = [analysis['size'] for analysis in cluster_analysis.values()]
    cluster_diversity = [analysis['mean_diversity'] for analysis in cluster_analysis.values()]
    cluster_heavy_ratio = [analysis['mean_heavy_ratio'] for analysis in cluster_analysis.values()]
    
    # Plot cluster sizes
    sns.barplot(x=range(n_clusters), y=cluster_sizes, ax=axes[0,0])
    axes[0,0].set_title('Cluster Sizes')
    axes[0,0].set_xlabel('Cluster')
    axes[0,0].set_ylabel('Number of Provinces')
    
    # Plot mean diversity index
    sns.barplot(x=range(n_clusters), y=cluster_diversity, ax=axes[0,1])
    axes[0,1].set_title('Mean Diversity Index by Cluster')
    axes[0,1].set_xlabel('Cluster')
    axes[0,1].set_ylabel('Mean Diversity Index')
    
    # Plot mean heavy vehicle ratio
    sns.barplot(x=range(n_clusters), y=cluster_heavy_ratio, ax=axes[1,0])
    axes[1,0].set_title('Mean Heavy Vehicle Ratio by Cluster')
    axes[1,0].set_xlabel('Cluster')
    axes[1,0].set_ylabel('Mean Heavy Vehicle Ratio')
    
    # Plot vehicle mix proportions for each cluster
    cluster_means = pd.DataFrame({
        cluster: analysis['mean_proportions']
        for cluster, analysis in cluster_analysis.items()
    }).T
    
    cluster_means.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Mean Vehicle Mix by Cluster')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from data_preparation import load_data, prepare_features
    
    # Load and prepare data
    df = load_data()
    df = prepare_features(df)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_k(df)
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # Perform clustering
    df, cluster_analysis = perform_kmeans_clustering(df, optimal_k)
    print("\nCluster Analysis:")
    for cluster, analysis in cluster_analysis.items():
        print(f"\nCluster {cluster}:")
        print(f"Size: {analysis['size']}")
        print(f"Locations: {', '.join(analysis['locations'])}")
        print(f"Mean Diversity Index: {analysis['mean_diversity']:.3f}")
        print(f"Mean Heavy Vehicle Ratio: {analysis['mean_heavy_ratio']:.3f}")
    
    # Plot cluster characteristics
    plot_cluster_characteristics(cluster_analysis, 'cluster_characteristics.png')
    
    # Detect anomalies
    df, anomalies = detect_anomalies(df)
    print("\nTop 3 Anomalous Provinces:")
    print(anomalies[['Location', 'Anomaly_Score']]) 