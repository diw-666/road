import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def plot_univariate_distributions(df, save_path=None):
    """Create histograms for total accidents and key proportions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total accidents distribution
    sns.histplot(data=df, x='Total_Accidents', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Total Accidents')
    
    # Motorcycle share distribution
    sns.histplot(data=df, x='Motor Cycle/Moped_Share', ax=axes[0,1])
    axes[0,1].set_title('Distribution of Motorcycle Share')
    
    # Heavy vehicle ratio distribution
    sns.histplot(data=df, x='Heavy_Vehicle_Ratio', ax=axes[1,0])
    axes[1,0].set_title('Distribution of Heavy Vehicle Ratio')
    
    # Diversity index distribution
    sns.histplot(data=df, x='Diversity_Index', ax=axes[1,1])
    axes[1,1].set_title('Distribution of Diversity Index')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_heatmap(df, save_path=None):
    """Create correlation heatmap of vehicle-mix proportions and derived metrics."""
    # Get relevant columns
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    metric_columns = ['Diversity_Index', 'Heavy_Vehicle_Ratio']
    columns_to_plot = share_columns + metric_columns
    
    # Compute correlation matrix
    corr_matrix = df[columns_to_plot].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Vehicle Mix and Derived Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def rank_provinces(df):
    """Rank provinces by various metrics."""
    rankings = {}
    
    # Total accidents ranking
    rankings['Total_Accidents'] = df.sort_values('Total_Accidents', ascending=False)[
        ['Location', 'Total_Accidents']
    ]
    
    # Motorcycle share ranking
    rankings['Motorcycle_Share'] = df.sort_values('Motor Cycle/Moped_Share', ascending=False)[
        ['Location', 'Motor Cycle/Moped_Share']
    ]
    
    # Bus share ranking (combining all bus types)
    bus_columns = ['SLT Bus', 'Private Bus', 'Intercity Bus']
    df['Total_Bus_Share'] = df[bus_columns].sum(axis=1) / df['Total_Accidents']
    rankings['Bus_Share'] = df.sort_values('Total_Bus_Share', ascending=False)[
        ['Location', 'Total_Bus_Share']
    ]
    
    # Diversity index ranking
    rankings['Diversity_Index'] = df.sort_values('Diversity_Index', ascending=False)[
        ['Location', 'Diversity_Index']
    ]
    
    return rankings

def perform_pca(df, n_components=2):
    """Perform PCA on vehicle-mix proportions."""
    # Get share columns
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[share_columns])
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_df['Location'] = df['Location'].values
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return pca_df, explained_variance, pca.components_

def plot_pca_results(pca_df, explained_variance, save_path=None):
    """Plot PCA results."""
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    
    # Add labels for each point
    for i, txt in enumerate(pca_df['Location']):
        plt.annotate(txt, (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]))
    
    plt.xlabel(f'PC1 ({explained_variance[0]:.1%} variance explained)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.1%} variance explained)')
    plt.title('PCA of Vehicle Mix Proportions')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    from data_preparation import load_data, prepare_features
    
    # Load and prepare data
    df = load_data()
    df = prepare_features(df)
    
    # Generate plots
    plot_univariate_distributions(df, 'univariate_distributions.png')
    plot_correlation_heatmap(df, 'correlation_heatmap.png')
    
    # Get rankings
    rankings = rank_provinces(df)
    print("\nTop 5 provinces by total accidents:")
    print(rankings['Total_Accidents'].head())
    
    # Perform PCA
    pca_df, explained_variance, components = perform_pca(df)
    plot_pca_results(pca_df, explained_variance, 'pca_results.png') 