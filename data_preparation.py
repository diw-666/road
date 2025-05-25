import pandas as pd
import numpy as np
from scipy.stats import entropy

def load_data(file_path='road_accident_data_by_vehicle_type.csv'):
    """Load and preprocess the accident data."""
    df = pd.read_csv(file_path)
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    df['Location'] = df['Location'].str.strip()
    return df

def compute_totals_and_rates(df):
    """Compute total accidents and vehicle-mix proportions for each province."""
    # Calculate total accidents per province
    df['Total_Accidents'] = df.iloc[:, 1:].sum(axis=1)
    
    # Calculate vehicle-mix proportions
    vehicle_columns = df.columns[1:-1]  # Exclude Location and Total_Accidents
    for col in vehicle_columns:
        df[f'{col}_Share'] = df[col] / df['Total_Accidents']
    
    return df

def compute_derived_metrics(df):
    """Compute diversity index and heavy-vehicle ratio."""
    # Get vehicle share columns
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    
    # Compute Shannon entropy (diversity index)
    df['Diversity_Index'] = df[share_columns].apply(
        lambda x: entropy(x, base=2), axis=1
    )
    
    # Compute heavy vehicle ratio
    heavy_vehicles = ['Lorry', 'Articulated Vehicle, prime mover', 'SLT Bus', 
                     'Private Bus', 'Intercity Bus']
    df['Heavy_Vehicle_Ratio'] = df[heavy_vehicles].sum(axis=1) / df['Total_Accidents']
    
    return df

def prepare_features(df):
    """Prepare all features for analysis."""
    df = compute_totals_and_rates(df)
    df = compute_derived_metrics(df)
    return df

if __name__ == "__main__":
    # Test the functions
    df = load_data()
    df = prepare_features(df)
    print("\nSample of processed data:")
    print(df[['Location', 'Total_Accidents', 'Diversity_Index', 'Heavy_Vehicle_Ratio']].head()) 