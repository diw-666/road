import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_preparation import load_data, prepare_features
from exploratory_analysis import perform_pca
from clustering_analysis import perform_kmeans_clustering, detect_anomalies
from predictive_modeling import prepare_modeling_data_improved, train_models_improved, create_vehicle_contribution_analysis

# Set page config
st.set_page_config(
    page_title="Road Accident Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #18191a !important;
        color: #f8f9fa !important;
    }
    .main {
        padding: 2rem;
        background-color: #18191a !important;
        color: #f8f9fa !important;
    }
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
        background-color: #23272f !important;
        color: #f8f9fa !important;
    }
    /* Metric cards */
    .stMetric {
        background-color: #23272f !important;
        color: #f8f9fa !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        border: none;
    }
    /* Plot containers */
    .stPlotlyChart {
        background-color: #23272f !important;
        color: #f8f9fa !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        border: none;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f8f9fa !important;
        font-weight: 600;
    }
    /* Dataframe styling */
    .dataframe, .stDataFrame {
        background-color: #23272f !important;
        color: #f8f9fa !important;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        border: none;
    }
    /* Button styling */
    .stButton>button {
        background-color: #1f77b4 !important;
        color: #f8f9fa !important;
        border: none;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1668a1 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    /* Selectbox styling */
    .stSelectbox, .stSlider {
        background-color: #23272f !important;
        color: #f8f9fa !important;
        border-radius: 0.3rem;
    }
    /* Remove white borders from all elements */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlock"] > div,
    div[data-testid="stVerticalBlock"] > div > div,
    div[data-testid="stVerticalBlock"] > div > div > div,
    div[data-testid="stVerticalBlock"] > div > div > div > div,
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stHorizontalBlock"] > div,
    div[data-testid="stHorizontalBlock"] > div > div,
    div[data-testid="stHorizontalBlock"] > div > div > div {
        border: none !important;
        background: none !important;
    }
    /* Remove white borders from plotly charts */
    .js-plotly-plot {
        border: none !important;
        background: none !important;
    }
    /* Remove white borders from metric containers */
    div[data-testid="stMetricValue"] {
        border: none !important;
        background: none !important;
        color: #f8f9fa !important;
    }
    /* Remove white borders from sidebar */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebarNav"] {
        border: none !important;
        background: #23272f !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_model_input(motor_car_share, motorcycle_share, three_wheeler_share, lorry_share, bus_share, cycle_share, other_share):
    """Create properly formatted input data for the accident rate prediction model using vehicle shares."""
    
    # Normalize shares to ensure they sum to 1
    total_share = motor_car_share + motorcycle_share + three_wheeler_share + lorry_share + bus_share + cycle_share + other_share
    if total_share == 0:
        total_share = 1  # Avoid division by zero
    
    # Normalize shares
    motor_car_norm = motor_car_share / total_share
    motorcycle_norm = motorcycle_share / total_share
    three_wheeler_norm = three_wheeler_share / total_share
    lorry_norm = lorry_share / total_share
    bus_norm = bus_share / total_share
    cycle_norm = cycle_share / total_share
    other_norm = other_share / total_share
    
    # Create input data with proper mapping for share-based features
    input_data = pd.DataFrame([{
        'Motor Car_Share': motor_car_norm,
        'Dual Purpose Vehicle_Share': other_norm * 0.4,  # Largest share of "other"
        'Lorry_Share': lorry_norm,
        'Cycle_Share': cycle_norm,
        'Motor Cycle/Moped_Share': motorcycle_norm,
        'Three wheeler_Share': three_wheeler_norm,
        'Articulated Vehicle, prime mover_Share': other_norm * 0.1,
        'SLT Bus_Share': other_norm * 0.1,
        'Private Bus_Share': bus_norm,
        'Intercity Bus_Share': other_norm * 0.1,
        'Land Vehicle/Tractor_Share': other_norm * 0.2,
        'Animal drawn vehicle or rider on animal_Share': other_norm * 0.05,
        'Other_Share': other_norm * 0.05
    }])
    
    # Add derived features that the model expects
    input_data['Diversity_Index'] = 1 - sum([s**2 for s in [motor_car_norm, motorcycle_norm, three_wheeler_norm, lorry_norm, bus_norm, cycle_norm, other_norm]])  # Simpson's diversity
    input_data['Heavy_Vehicle_Ratio'] = lorry_norm + bus_norm + (other_norm * 0.2)  # Include some heavy vehicles from "other"
    
    # Add interaction features based on shares
    input_data['Motor_Vehicle_Share_Total'] = motor_car_norm + motorcycle_norm + three_wheeler_norm
    input_data['Commercial_Vehicle_Share_Total'] = lorry_norm + bus_norm + (other_norm * 0.2)  # Include some commercial from "other"
    input_data['Heavy_Light_Share_Ratio'] = input_data['Commercial_Vehicle_Share_Total'] / (input_data['Motor_Vehicle_Share_Total'] + 0.01)
    
    # Add squared terms for non-linear relationships
    share_columns = [col for col in input_data.columns if col.endswith('_Share')]
    for col in share_columns:
        input_data[f'{col}_squared'] = input_data[col] ** 2
    
    return input_data

def analyze_vehicle_contributions(model, feature_names, input_shares, prediction):
    """Analyze how each vehicle type contributes to the predicted accident rate."""
    
    # Create input shares dictionary for analysis
    shares_dict = {}
    vehicle_types = ['Motor Car', 'Motor Cycle/Moped', 'Three wheeler', 'Lorry', 'Private Bus', 'Cycle', 'Other']
    input_values = [input_shares.get(f'{v}_Share', 0) for v in vehicle_types]
    
    for vehicle, value in zip(vehicle_types, input_values):
        shares_dict[f'{vehicle}_Share'] = value
    
    # Get contributions
    contributions = create_vehicle_contribution_analysis(model, feature_names, shares_dict)
    
    # Calculate relative contributions
    total_contribution = sum([c['contribution'] for c in contributions.values()])
    if total_contribution > 0:
        for vehicle in contributions:
            contributions[vehicle]['relative_contribution'] = contributions[vehicle]['contribution'] / total_contribution
    
    return contributions

def main():
    st.title("🚗 Road Accident Analysis Dashboard")
    st.markdown("""
    This dashboard provides a comprehensive analysis of road accidents by vehicle type across provinces in Sri Lanka.
    Use the sidebar to navigate between different analysis sections.
    """)
    
    # Load and prepare data
    with st.spinner('Loading and preparing data...'):
        df = load_data()
        df = prepare_features(df)
    
    # Sidebar navigation with descriptions
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Choose Analysis Section",
        ["Overview", "Province Profiles", "Clustering Analysis", "Predictive Modeling"],
        format_func=lambda x: {
            "Overview": "📊 Overview & Key Metrics",
            "Province Profiles": "🏢 Province-Specific Analysis",
            "Clustering Analysis": "🔍 Pattern Discovery",
            "Predictive Modeling": "📈 Accident Prediction"
        }[x]
    )
    
    # Add data info to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.markdown(f"Total Provinces: {len(df)}")
    st.sidebar.markdown(f"Total Accidents: {df['Total_Accidents'].sum():,.0f}")
    
    if page == "Overview":
        show_overview(df)
    elif page == "Province Profiles":
        show_province_profiles(df)
    elif page == "Clustering Analysis":
        show_clustering_analysis(df)
    else:
        show_predictive_modeling(df)

def show_overview(df):
    st.header("📊 Overview of Road Accidents")
    st.markdown("""
    **What is this?**  
    This section provides a high-level overview of road accidents across Sri Lanka. You can see the total number of accidents, which provinces are most affected, and the diversity of vehicle types involved.
    
    **Why it matters:**  
    Understanding the overall distribution helps identify hotspots and trends, guiding resource allocation and policy decisions.
    """)
    
    # Key metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Accidents",
            f"{df['Total_Accidents'].sum():,.0f}",
            "Across all provinces"
        )
        st.caption("**Analysis:** This is the total number of reported road accidents in the dataset. A high number may indicate systemic issues or reporting efficiency.")
    with col2:
        st.metric(
            "Highest Accident Province",
            df.loc[df['Total_Accidents'].idxmax(), 'Location'],
            f"{df['Total_Accidents'].max():,.0f} accidents"
        )
        st.caption(f"**Analysis:** {df.loc[df['Total_Accidents'].idxmax(), 'Location']} has the highest accident count, suggesting it may need targeted interventions.")
    with col3:
        st.metric(
            "Highest Motorcycle Share",
            df.loc[df['Motor Cycle/Moped_Share'].idxmax(), 'Location'],
            f"{df['Motor Cycle/Moped_Share'].max():.1%}"
        )
        st.caption(f"**Analysis:** {df.loc[df['Motor Cycle/Moped_Share'].idxmax(), 'Location']} has the largest proportion of motorcycle-related accidents, indicating a focus area for two-wheeler safety.")
    with col4:
        st.metric(
            "Highest Diversity Index",
            df.loc[df['Diversity_Index'].idxmax(), 'Location'],
            f"{df['Diversity_Index'].max():.2f}"
        )
        st.caption(f"**Analysis:** {df.loc[df['Diversity_Index'].idxmax(), 'Location']} has the most diverse mix of vehicle types in accidents, which may reflect complex traffic patterns.")
    
    st.subheader("Total Accidents by Province")
    st.markdown("""
    **What is this?**  
    This bar chart shows the number of accidents in each province.
    
    **Why it matters:**  
    Provinces with higher accident counts may require more attention from policymakers and law enforcement.
    """)
    fig = px.bar(
        df.sort_values('Total_Accidents', ascending=False),
        x='Location',
        y='Total_Accidents',
        title='Total Accidents by Province',
        labels={'Location': 'Province', 'Total_Accidents': 'Number of Accidents'},
        color='Total_Accidents',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** Colombo and a few other provinces dominate the accident statistics, highlighting regional disparities.")
    
    st.subheader("Vehicle Mix Patterns")
    st.markdown("""
    **What is this?**  
    The heatmap below shows the proportion of different vehicle types involved in accidents for each province. Darker colors indicate higher proportions.
    
    **Why it matters:**  
    This helps identify which vehicle types are most at risk in each region, guiding targeted safety campaigns.
    """)
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    fig = px.imshow(
        df[share_columns].T,
        labels=dict(x="Province", y="Vehicle Type", color="Share"),
        x=df['Location'],
        title='Vehicle Mix Heatmap by Province',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** Some provinces have a high share of motorcycle accidents, while others see more heavy vehicle involvement.")

def show_province_profiles(df):
    st.header("🏢 Province Profiles")
    st.markdown("""
    **What is this?**  
    Select a province to view its detailed accident profile, including vehicle mix distribution and comparison with other provinces.
    
    **Why it matters:**  
    Province-level analysis helps identify local risk factors and tailor interventions to specific regions.
    """)
    
    # Province selector with search
    selected_province = st.selectbox(
        "Select Province",
        df['Location'].unique(),
        format_func=lambda x: f"📍 {x}"
    )
    
    # Get province data
    province_data = df[df['Location'] == selected_province].iloc[0]
    
    # Key metrics in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Accidents",
            f"{province_data['Total_Accidents']:,.0f}",
            f"Rank: #{df['Total_Accidents'].rank(ascending=False)[df['Location'] == selected_province].iloc[0]:.0f}"
        )
        st.caption("**Analysis:** Shows the total number of accidents in this province and its rank compared to others. High values may indicate greater risk or more traffic.")
    with col2:
        st.metric(
            "Diversity Index",
            f"{province_data['Diversity_Index']:.2f}",
            "Higher values indicate more diverse vehicle mix"
        )
        st.caption("**Analysis:** A higher diversity index means accidents involve a wider range of vehicle types, which may reflect complex traffic patterns.")
    with col3:
        st.metric(
            "Heavy Vehicle Ratio",
            f"{province_data['Heavy_Vehicle_Ratio']:.1%}",
            "Proportion of heavy vehicles"
        )
        st.caption("**Analysis:** Indicates the share of heavy vehicles in accidents. High ratios may suggest a need for heavy vehicle safety measures.")
    
    # Vehicle mix pie chart
    st.subheader("Vehicle Mix Distribution")
    st.markdown("""
    **What is this?**  
    This pie chart shows the proportion of each vehicle type involved in accidents in the selected province.
    
    **Why it matters:**  
    Understanding the vehicle mix helps target safety campaigns and regulations to the most at-risk groups.
    """)
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    vehicle_mix = pd.DataFrame({
        'Vehicle Type': [col.replace('_Share', '').replace('_', ' ').title() 
                        for col in share_columns],
        'Share': [province_data[col] for col in share_columns]
    })
    
    fig = px.pie(
        vehicle_mix,
        values='Share',
        names='Vehicle Type',
        title=f'Vehicle Mix in {selected_province}',
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** The largest slice shows the most common vehicle type in accidents here. Compare with other provinces to spot unique risks.")
    
    # Comparison with other provinces
    st.subheader("Province Comparison")
    st.markdown("""
    **What is this?**  
    This bar chart compares the selected province to others using the chosen metric. The red marker shows the selected province's position.
    
    **Why it matters:**  
    Comparing provinces helps identify outliers and best practices.
    """)
    
    metric = st.selectbox(
        "Select Metric to Compare",
        ['Total_Accidents', 'Diversity_Index', 'Heavy_Vehicle_Ratio'],
        format_func=lambda x: {
            'Total_Accidents': 'Total Accidents',
            'Diversity_Index': 'Diversity Index',
            'Heavy_Vehicle_Ratio': 'Heavy Vehicle Ratio'
        }[x]
    )
    
    fig = px.bar(
        df.sort_values(metric, ascending=False),
        x='Location',
        y=metric,
        title=f'{metric.replace("_", " ").title()} by Province',
        labels={'Location': 'Province', metric: metric.replace('_', ' ').title()},
        color=metric,
        color_continuous_scale='Viridis'
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_province],
            y=[province_data[metric]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Selected Province'
        )
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"**Analysis:** {selected_province} is highlighted. If it's at the top or bottom, it may be an outlier for this metric.")

def show_clustering_analysis(df):
    st.header("🔍 Pattern Discovery")
    st.markdown("""
    **What is this?**  
    This section uses machine learning to discover patterns in accident data. Provinces are grouped by similarity in vehicle mix patterns.
    
    **Why it matters:**  
    Clustering helps identify provinces with similar risk profiles, which can inform regional policy and resource allocation.
    """)
    
    # Perform clustering
    with st.spinner('Performing clustering analysis...'):
        optimal_k = 4  # You might want to compute this dynamically
        df_clustered, cluster_analysis = perform_kmeans_clustering(df, optimal_k)
    
    # PCA visualization
    st.subheader("Province Similarity Map")
    st.markdown("""
    **What is this?**  
    This scatter plot uses Principal Component Analysis (PCA) to show how similar provinces are based on their vehicle mix patterns. Each point is a province, colored by cluster.
    
    **Why it matters:**  
    Provinces close together have similar accident profiles. Clusters may reveal shared challenges or opportunities.
    """)
    pca_df, explained_variance, _ = perform_pca(df)
    pca_df['Cluster'] = df_clustered['Cluster'].astype(str)
    
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_name='Location',
        title='Province Similarity Map (PCA)',
        labels={
            'PC1': f'Principal Component 1 ({explained_variance[0]:.1%} variance)',
            'PC2': f'Principal Component 2 ({explained_variance[1]:.1%} variance)'
        }
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** Provinces in the same cluster may benefit from similar interventions. Outliers may need special attention.")
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    st.markdown("""
    **What is this?**  
    Select a cluster to view its characteristics and member provinces. The bar chart shows the average vehicle mix for the cluster.
    
    **Why it matters:**  
    Understanding cluster profiles helps tailor safety strategies to groups of provinces with similar risks.
    """)
    selected_cluster = st.selectbox(
        "Select Cluster",
        range(optimal_k),
        format_func=lambda x: f"Cluster {x+1}"
    )
    cluster_data = cluster_analysis[selected_cluster]
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Number of Provinces",
            cluster_data['size'],
            "in this cluster"
        )
        st.caption("**Analysis:** Shows how many provinces share similar accident patterns in this cluster.")
    with col2:
        st.metric(
            "Mean Diversity Index",
            f"{cluster_data['mean_diversity']:.2f}",
            "Average diversity of vehicle mix"
        )
        st.caption("**Analysis:** Higher values mean more varied vehicle types in cluster accidents.")
    st.markdown("### Member Provinces")
    st.markdown(", ".join([f"**{loc}**" for loc in cluster_data['locations']]))
    st.caption("**Analysis:** These provinces share similar accident characteristics.")
    st.subheader("Cluster Characteristics")
    fig = px.bar(
        pd.DataFrame({
            'Vehicle Type': [col.replace('_Share', '').replace('_', ' ').title() 
                           for col in df.columns if col.endswith('_Share')],
            'Mean Share': cluster_data['mean_proportions']
        }),
        x='Vehicle Type',
        y='Mean Share',
        title=f'Average Vehicle Mix for Cluster {selected_cluster+1}',
        labels={'Mean Share': 'Average Proportion', 'Vehicle Type': 'Type of Vehicle'}
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** The tallest bars show which vehicle types are most involved in accidents for this cluster.")
    # Anomaly detection
    st.subheader("Anomaly Detection")
    st.markdown("""
    **What is this?**  
    This scatter plot highlights provinces with unusual accident patterns (anomalies). Higher anomaly scores mean more unusual patterns.
    
    **Why it matters:**  
    Anomalies may indicate data quality issues or unique local factors needing further investigation.
    """)
    df_anomalies, anomalies = detect_anomalies(df)
    fig = px.scatter(
        df_anomalies,
        x='Diversity_Index',
        y='Heavy_Vehicle_Ratio',
        hover_name='Location',
        color='Anomaly_Score',
        title='Anomaly Detection Results',
        labels={
            'Diversity_Index': 'Diversity Index',
            'Heavy_Vehicle_Ratio': 'Heavy Vehicle Ratio',
            'Anomaly_Score': 'Anomaly Score'
        },
        color_continuous_scale='Viridis'
    )
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** Provinces with the highest anomaly scores may require special attention or further study.")
    st.markdown("### Top 3 Most Unusual Provinces")
    st.dataframe(
        anomalies[['Location', 'Anomaly_Score']].rename(
            columns={'Location': 'Province', 'Anomaly_Score': 'Unusualness Score'}
        ),
        hide_index=True
    )
    st.caption("**Analysis:** These are the most unusual provinces based on accident patterns.")

def show_predictive_modeling(df):
    st.header("📈 Accident Prediction")
    st.markdown("""
    **What is this?**  
    This section uses machine learning to predict accident rates (per 1000 vehicles) based on vehicle mix patterns. You can compare model performance and make predictions for different scenarios.
    
    **Why it matters:**  
    Predictive modeling helps estimate accident risk rates and evaluate the impact of changes in vehicle mix or policy interventions.
    """)
    # Prepare modeling data
    with st.spinner('Preparing and training models...'):
        X, y = prepare_modeling_data_improved(df)
        results, X_test, y_test = train_models_improved(X, y)
    st.subheader("Model Performance Comparison")
    st.markdown("""
    **What is this?**  
    This table compares the performance of different prediction models using RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and R² Score.
    
    **Why it matters:**  
    Lower RMSE and MAE values indicate better predictions. R² shows how much variance in accidents is explained by the model.
    """)
    metrics_df = pd.DataFrame({
        model: result['metrics']
        for model, result in results.items()
    }).T
    metrics_df = metrics_df.round(2)
    metrics_df.columns = ['Root Mean Square Error', 'Mean Absolute Error', 'R² Score']
    st.dataframe(
        metrics_df.style.background_gradient(cmap='RdYlGn_r', subset=['R² Score'])
        .background_gradient(cmap='RdYlGn', subset=['Root Mean Square Error', 'Mean Absolute Error']),
        use_container_width=True
    )
    st.caption("**Analysis:** The best model has the lowest RMSE and MAE, and the highest R². Use this model for predictions.")
    st.subheader("Feature Importance")
    st.markdown("""
    **What is this?**  
    This bar chart shows which factors (vehicle types or derived metrics) most strongly influence accident rates in the best model.
    
    **Why it matters:**  
    Features with higher importance have a greater impact on accident rate predictions. Focus on these for interventions.
    """)
    best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
    best_model = results[best_model_name]['model']
    importance_df = pd.DataFrame({
        'Feature': [name.replace('_Share', '').replace('_', ' ').title() 
                   for name in X.columns],
        'Importance': best_model.feature_importances_ if hasattr(best_model, 'feature_importances_')
        else np.abs(best_model.coef_)
    }).sort_values('Importance', ascending=False)
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        title=f'Feature Importance ({best_model_name})',
        labels={'Importance': 'Relative Importance', 'Feature': 'Factor'},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Analysis:** The top features are the most influential in predicting accident rates. Prioritize these for safety improvements.")
    st.subheader("🎯 Interactive Accident Prediction Tool")
    
    # Add comprehensive explanation
    st.markdown("""
    ### 📋 How This Tool Works
    
    **🎯 What it predicts:** Accident rate (accidents per 1000 vehicles) in a province/region
    
    **📊 What you input:** Share/percentage of each vehicle type in the region (0.0 to 1.0)
    
    **🔄 How it works:** The model estimates accident risk rate based on vehicle mix proportions and shows which vehicle types contribute most to accidents
    """)
    
    # Create two main sections: Quick Scenarios and Custom Input
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["🚀 Quick Scenarios", "🎚️ Interactive Sliders", "⚙️ Custom Input"],
        horizontal=True
    )
    
    if prediction_mode == "🚀 Quick Scenarios":
        st.markdown("### 🚀 Quick Vehicle Mix Scenarios")
        st.markdown("Select a realistic vehicle mix scenario to predict accident rate:")
        
        # Define realistic vehicle mix scenarios (as shares)
        scenarios = {
            "Small Rural Province": {
                "description": "Small rural province with typical rural vehicle mix (like Mulathivu or Mannar)",
                "icon": "🌾",
                "Motor Car": 0.15,
                "Motor Cycle/Moped": 0.45,
                "Three wheeler": 0.10,
                "Lorry": 0.08,
                "Private Bus": 0.02,
                "Cycle": 0.15,
                "Other vehicles": 0.05
            },
            "Medium Province": {
                "description": "Medium-sized province with balanced vehicle mix (like Matale or Polonnaruwa)",
                "icon": "🏘️",
                "Motor Car": 0.20,
                "Motor Cycle/Moped": 0.40,
                "Three wheeler": 0.12,
                "Lorry": 0.08,
                "Private Bus": 0.03,
                "Cycle": 0.12,
                "Other vehicles": 0.05
            },
            "Large Urban Province": {
                "description": "Major urban center with diverse vehicle mix (like Kandy or Galle)",
                "icon": "🏙️",
                "Motor Car": 0.25,
                "Motor Cycle/Moped": 0.35,
                "Three wheeler": 0.15,
                "Lorry": 0.08,
                "Private Bus": 0.05,
                "Cycle": 0.08,
                "Other vehicles": 0.04
            },
            "Major Metropolitan": {
                "description": "Largest metropolitan area with car-heavy mix (like Colombo)",
                "icon": "🌆",
                "Motor Car": 0.35,
                "Motor Cycle/Moped": 0.30,
                "Three wheeler": 0.15,
                "Lorry": 0.06,
                "Private Bus": 0.08,
                "Cycle": 0.04,
                "Other vehicles": 0.02
            }
        }
        
        # Scenario selection
        selected_scenario = st.selectbox(
            "Choose a scenario:",
            list(scenarios.keys()),
            format_func=lambda x: f"{scenarios[x]['icon']} {x}"
        )
        
        scenario_data = scenarios[selected_scenario]
        st.info(f"**{scenario_data['description']}**")
        
        # Show scenario details in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Vehicle Mix (Shares):**")
            for vehicle, share in scenario_data.items():
                if vehicle not in ['description', 'icon']:
                    st.markdown(f"• {vehicle}: **{share:.1%}**")
        
        with col2:
            # Calculate total share for this scenario
            total_share = sum(share for key, share in scenario_data.items() 
                            if key not in ['description', 'icon'])
            st.metric("Total Share", f"{total_share:.1%}")
            
            # Show diversity
            diversity = len([v for k, v in scenario_data.items() if k not in ['description', 'icon'] and v > 0.05])
            st.metric("Vehicle Diversity", f"{diversity} main types")
        
        # Predict button
        if st.button("🔮 Predict Accident Rate", type="primary", key="scenario_predict"):
            with st.spinner('Calculating accident rate prediction based on vehicle mix...'):
                # Create input data matching the model's expected features
                input_data = create_model_input(
                    scenario_data['Motor Car'],
                    scenario_data['Motor Cycle/Moped'],
                    scenario_data['Three wheeler'],
                    scenario_data['Lorry'],
                    scenario_data['Private Bus'],
                    scenario_data['Cycle'],
                    scenario_data['Other vehicles']
                )
                
                # Make prediction
                scaler = results[best_model_name]['scaler']
                input_data_scaled = scaler.transform(input_data)
                prediction = best_model.predict(input_data_scaled)[0]
                cv_std = results[best_model_name]['cv_std']
                
                # Analyze vehicle contributions
                contributions = analyze_vehicle_contributions(
                    best_model, 
                    list(input_data.columns), 
                    input_data.iloc[0].to_dict(), 
                    prediction
                )
                
                # Display results
                st.markdown("### 🎯 Accident Rate Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "🎯 Predicted Accident Rate",
                        f"{prediction:.1f}",
                        f"±{1.96 * cv_std:.1f} per 1000 vehicles"
                    )
                with col2:
                    # Show highest contributing vehicle type
                    if contributions:
                        top_contributor = max(contributions.items(), key=lambda x: x[1].get('contribution', 0))
                        st.metric(
                            "🚨 Highest Risk Vehicle",
                            top_contributor[0],
                            f"{top_contributor[1].get('relative_contribution', 0):.1%} contribution"
                        )
                with col3:
                    # Risk assessment
                    if prediction < 15:
                        risk_level = "🟢 Low Risk"
                    elif prediction < 25:
                        risk_level = "🟡 Medium Risk"
                    else:
                        risk_level = "🔴 High Risk"
                    st.metric("🚨 Risk Assessment", risk_level)
                
                # Show vehicle contributions
                if contributions:
                    st.markdown("### 📊 Vehicle Type Contributions to Accident Rate")
                    contrib_data = []
                    for vehicle, data in contributions.items():
                        contrib_data.append({
                            'Vehicle Type': vehicle,
                            'Share in Mix': f"{data.get('share', 0):.1%}",
                            'Risk Factor': f"{data.get('risk_factor', 1.0):.1f}x",
                            'Contribution': f"{data.get('relative_contribution', 0):.1%}",
                            'Impact Level': 'High' if data.get('relative_contribution', 0) > 0.2 else 'Medium' if data.get('relative_contribution', 0) > 0.1 else 'Low'
                        })
                    
                    contrib_df = pd.DataFrame(contrib_data)
                    contrib_df = contrib_df.sort_values('Contribution', ascending=False)
                    st.dataframe(contrib_df, hide_index=True)
                    
                    # Create contribution chart
                    fig = px.bar(
                        contrib_df.head(6),  # Top 6 contributors
                        x='Vehicle Type',
                        y=[float(x.strip('%'))/100 for x in contrib_df.head(6)['Contribution']],
                        title='Vehicle Type Contributions to Accident Rate',
                        labels={'y': 'Contribution to Accident Rate', 'x': 'Vehicle Type'},
                        color=[float(x.strip('%'))/100 for x in contrib_df.head(6)['Contribution']],
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                st.markdown("### 💡 Understanding the Results")
                st.info(f"""
                **How the prediction works:**
                - **Accident rate** shows accidents per 1000 vehicles in the region
                - **Vehicle contributions** show which types drive accident risk most
                - **Risk factors** are based on historical accident patterns per vehicle type
                - **Higher contributions** indicate vehicle types that need more safety focus
                - **Mixed traffic** with diverse vehicle types can increase complexity
                """)
    
    elif prediction_mode == "🎚️ Interactive Sliders":
        st.markdown("### 🎚️ Auto-Normalizing Vehicle Mix Sliders")
        st.markdown("Adjust any slider - the others will automatically adjust to keep the total at 100%:")
        
        # Initialize session state for sliders if not exists
        if 'slider_values' not in st.session_state:
            st.session_state.slider_values = {
                'motor_car': 25.0,
                'motorcycle': 35.0,
                'three_wheeler': 15.0,
                'lorry': 8.0,
                'bus': 5.0,
                'cycle': 8.0,
                'other': 4.0
            }
        
        # Create sliders with auto-normalization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            motor_car_pct = st.slider(
                "🚗 Motor Cars (%)",
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['motor_car'], 
                step=1.0, format="%.0f%%",
                key="motor_car_slider",
                help="Percentage of motor cars in vehicle mix"
            )
            motorcycle_pct = st.slider(
                "🏍️ Motorcycles/Mopeds (%)", 
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['motorcycle'], 
                step=1.0, format="%.0f%%",
                key="motorcycle_slider",
                help="Percentage of motorcycles and mopeds"
            )
            three_wheeler_pct = st.slider(
                "🛺 Three Wheelers (%)",
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['three_wheeler'], 
                step=1.0, format="%.0f%%",
                key="three_wheeler_slider",
                help="Percentage of three wheelers"
            )
        
        with col2:
            lorry_pct = st.slider(
                "🚛 Lorries (%)",
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['lorry'], 
                step=1.0, format="%.0f%%",
                key="lorry_slider",
                help="Percentage of lorries"
            )
            bus_pct = st.slider(
                "🚌 Private Buses (%)",
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['bus'], 
                step=1.0, format="%.0f%%",
                key="bus_slider",
                help="Percentage of private buses"
            )
            cycle_pct = st.slider(
                "🚴 Bicycles (%)",
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['cycle'], 
                step=1.0, format="%.0f%%",
                key="cycle_slider",
                help="Percentage of bicycles"
            )
        
        with col3:
            other_pct = st.slider(
                "🚙 Other Vehicles (%)",
                min_value=0.0, max_value=100.0, 
                value=st.session_state.slider_values['other'], 
                step=1.0, format="%.0f%%",
                key="other_slider",
                help="Percentage of other vehicles"
            )
            
            # Calculate and show total
            total_pct = motor_car_pct + motorcycle_pct + three_wheeler_pct + lorry_pct + bus_pct + cycle_pct + other_pct
            
            # Color-code the total based on how close to 100%
            if abs(total_pct - 100) < 1:
                st.success(f"✅ Total: {total_pct:.0f}%")
            elif abs(total_pct - 100) < 5:
                st.warning(f"⚠️ Total: {total_pct:.0f}%")
            else:
                st.error(f"❌ Total: {total_pct:.0f}%")
            
            # Auto-normalize button
            if st.button("🎯 Auto-Normalize to 100%", help="Automatically adjust all values to sum to 100%"):
                if total_pct > 0:
                    # Normalize all values proportionally
                    factor = 100.0 / total_pct
                    st.session_state.slider_values = {
                        'motor_car': motor_car_pct * factor,
                        'motorcycle': motorcycle_pct * factor,
                        'three_wheeler': three_wheeler_pct * factor,
                        'lorry': lorry_pct * factor,
                        'bus': bus_pct * factor,
                        'cycle': cycle_pct * factor,
                        'other': other_pct * factor
                    }
                    st.rerun()
        
        # Show current mix as a pie chart
        if total_pct > 0:
            st.markdown("### 📊 Current Vehicle Mix")
            mix_data = pd.DataFrame({
                'Vehicle Type': ['Motor Cars', 'Motorcycles', 'Three Wheelers', 'Lorries', 'Buses', 'Bicycles', 'Other'],
                'Percentage': [motor_car_pct, motorcycle_pct, three_wheeler_pct, lorry_pct, bus_pct, cycle_pct, other_pct]
            })
            
            fig = px.pie(
                mix_data,
                values='Percentage',
                names='Vehicle Type',
                title=f'Vehicle Mix Distribution (Total: {total_pct:.0f}%)',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Convert percentages to shares for prediction
        motor_car_share = motor_car_pct / 100.0
        motorcycle_share = motorcycle_pct / 100.0
        three_wheeler_share = three_wheeler_pct / 100.0
        lorry_share = lorry_pct / 100.0
        bus_share = bus_pct / 100.0
        cycle_share = cycle_pct / 100.0
        other_share = other_pct / 100.0
        total_share = total_pct / 100.0
        
        # Predict button for interactive sliders
        if st.button("🔮 Predict Accident Rate", type="primary", key="interactive_predict"):
            if total_share == 0:
                st.error("Please enter at least some vehicle shares!")
            else:
                with st.spinner('Calculating accident rate prediction based on vehicle mix...'):
                    # Create input data for the model
                    input_data = create_model_input(
                        motor_car_share,
                        motorcycle_share,
                        three_wheeler_share,
                        lorry_share,
                        bus_share,
                        cycle_share,
                        other_share
                    )
                    
                    # Make prediction
                    scaler = results[best_model_name]['scaler']
                    input_data_scaled = scaler.transform(input_data)
                    prediction = best_model.predict(input_data_scaled)[0]
                    cv_std = results[best_model_name]['cv_std']
                    
                    # Analyze vehicle contributions
                    contributions = analyze_vehicle_contributions(
                        best_model, 
                        list(input_data.columns), 
                        input_data.iloc[0].to_dict(), 
                        prediction
                    )
                    
                    # Display results
                    st.markdown("### 🎯 Interactive Accident Rate Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "🎯 Predicted Accident Rate",
                            f"{prediction:.1f}",
                            f"±{1.96 * cv_std:.1f} per 1000 vehicles"
                        )
                    with col2:
                        # Show normalized motorcycle share
                        normalized_motorcycle = motorcycle_share / total_share if total_share > 0 else 0
                        st.metric(
                            "🏍️ Motorcycle Share",
                            f"{normalized_motorcycle:.1%}",
                            "after normalization"
                        )
                    with col3:
                        model_confidence = results[best_model_name]['metrics']['R2']
                        st.metric(
                            "🎯 Model Accuracy",
                            f"{model_confidence:.1%}",
                            "R² Score"
                        )
                    
                    # Show vehicle contributions
                    if contributions:
                        st.markdown("### 📊 Vehicle Type Contributions to Accident Rate")
                        contrib_data = []
                        for vehicle, data in contributions.items():
                            contrib_data.append({
                                'Vehicle Type': vehicle,
                                'Share in Mix': f"{data.get('share', 0):.1%}",
                                'Risk Factor': f"{data.get('risk_factor', 1.0):.1f}x",
                                'Contribution': f"{data.get('relative_contribution', 0):.1%}",
                                'Impact Level': 'High' if data.get('relative_contribution', 0) > 0.2 else 'Medium' if data.get('relative_contribution', 0) > 0.1 else 'Low'
                            })
                        
                        contrib_df = pd.DataFrame(contrib_data)
                        contrib_df = contrib_df.sort_values('Contribution', ascending=False)
                        st.dataframe(contrib_df, hide_index=True)
                        
                        # Create contribution chart
                        fig = px.bar(
                            contrib_df.head(6),  # Top 6 contributors
                            x='Vehicle Type',
                            y=[float(x.strip('%'))/100 for x in contrib_df.head(6)['Contribution']],
                            title='Vehicle Type Contributions to Accident Rate',
                            labels={'y': 'Contribution to Accident Rate', 'x': 'Vehicle Type'},
                            color=[float(x.strip('%'))/100 for x in contrib_df.head(6)['Contribution']],
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Provide insights
                    st.markdown("### 🔍 Analysis")
                    
                    # Risk assessment based on vehicle mix
                    normalized_shares = [s/total_share for s in [motor_car_share, motorcycle_share, three_wheeler_share, lorry_share, bus_share, cycle_share, other_share]]
                    motorcycle_norm = normalized_shares[1]  # motorcycle share
                    heavy_vehicle_norm = normalized_shares[3] + normalized_shares[4]  # lorry + bus
                    
                    if prediction > 25:
                        st.error(f"🚨 **High accident rate** ({prediction:.1f} per 1000 vehicles). Urgent safety interventions needed.")
                    elif prediction > 15:
                        st.warning(f"⚠️ **Moderate accident rate** ({prediction:.1f} per 1000 vehicles). Consider targeted safety measures.")
                    else:
                        st.success(f"✅ **Low accident rate** ({prediction:.1f} per 1000 vehicles). Relatively safe traffic environment.")
                    
                    if motorcycle_norm > 0.4:
                        st.warning(f"⚠️ **High motorcycle share** ({motorcycle_norm:.1%}). This significantly increases accident risk due to motorcycle vulnerability.")
                    elif motorcycle_norm > 0.25:
                        st.info(f"ℹ️ **Moderate motorcycle share** ({motorcycle_norm:.1%}). Consider motorcycle safety measures.")
                    
                    if heavy_vehicle_norm > 0.15:
                        st.warning(f"🚛 **High heavy vehicle share** ({heavy_vehicle_norm:.1%}). Focus on heavy vehicle safety and traffic management.")
                    
                    # Show top risk contributor
                    if contributions:
                        top_contributor = max(contributions.items(), key=lambda x: x[1].get('contribution', 0))
                        st.info(f"🎯 **Primary risk factor:** {top_contributor[0]} contributes {top_contributor[1].get('relative_contribution', 0):.1%} to the accident rate.")

    else:  # Custom Input mode
        st.markdown("### ⚙️ Custom Vehicle Mix Input")
        st.markdown("Enter the share/percentage of each vehicle type in your region:")
        
        st.info("""
        💡 **Input Guide:** Enter values between 0.0 and 1.0 representing the proportion of each vehicle type. 
        For example, 0.3 means 30% of vehicles are of that type. Values don't need to sum to exactly 1.0.
        """)
        
        # Create input form for main vehicle types
        col1, col2, col3 = st.columns(3)
        
        with col1:
            motor_car_share = st.number_input(
                "🚗 Motor Cars",
                min_value=0.0, max_value=1.0, value=0.25, step=0.01, format="%.2f",
                help="Share of motor cars (0.0 = 0%, 1.0 = 100%)"
            )
            motorcycle_share = st.number_input(
                "🏍️ Motorcycles/Mopeds", 
                min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.2f",
                help="Share of motorcycles and mopeds"
            )
            three_wheeler_share = st.number_input(
                "🛺 Three Wheelers",
                min_value=0.0, max_value=1.0, value=0.15, step=0.01, format="%.2f",
                help="Share of three wheelers"
            )
        
        with col2:
            lorry_share = st.number_input(
                "🚛 Lorries",
                min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.2f",
                help="Share of lorries"
            )
            bus_share = st.number_input(
                "🚌 Private Buses",
                min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f",
                help="Share of private buses"
            )
            cycle_share = st.number_input(
                "🚴 Bicycles",
                min_value=0.0, max_value=1.0, value=0.08, step=0.01, format="%.2f",
                help="Share of bicycles"
            )
        
        with col3:
            other_share = st.number_input(
                "🚙 Other Vehicles",
                min_value=0.0, max_value=1.0, value=0.04, step=0.01, format="%.2f",
                help="Share of other vehicles (dual purpose, tractors, etc.)"
            )
            
            # Show total share
            total_share = motor_car_share + motorcycle_share + three_wheeler_share + lorry_share + bus_share + cycle_share + other_share
            st.metric("📊 Total Share", f"{total_share:.1%}")
            
            # Show normalization info
            if total_share > 0:
                st.caption(f"Shares will be normalized to 100%")
        
        # Predict button for custom input
        if st.button("🔮 Predict Accident Rate", type="primary", key="custom_predict"):
            if total_share == 0:
                st.error("Please enter at least some vehicle shares!")
            else:
                with st.spinner('Calculating accident rate prediction based on vehicle mix...'):
                    # Create input data for the model
                    input_data = create_model_input(
                        motor_car_share,
                        motorcycle_share,
                        three_wheeler_share,
                        lorry_share,
                        bus_share,
                        cycle_share,
                        other_share
                    )
                    
                    # Make prediction
                    scaler = results[best_model_name]['scaler']
                    input_data_scaled = scaler.transform(input_data)
                    prediction = best_model.predict(input_data_scaled)[0]
                    cv_std = results[best_model_name]['cv_std']
                    
                    # Analyze vehicle contributions
                    contributions = analyze_vehicle_contributions(
                        best_model, 
                        list(input_data.columns), 
                        input_data.iloc[0].to_dict(), 
                        prediction
                    )
                    
                    # Display results
                    st.markdown("### 🎯 Custom Accident Rate Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "🎯 Predicted Accident Rate",
                            f"{prediction:.1f}",
                            f"±{1.96 * cv_std:.1f} per 1000 vehicles"
                        )
                    with col2:
                        # Show normalized motorcycle share
                        normalized_motorcycle = motorcycle_share / total_share if total_share > 0 else 0
                        st.metric(
                            "🏍️ Motorcycle Share",
                            f"{normalized_motorcycle:.1%}",
                            "after normalization"
                        )
                    with col3:
                        model_confidence = results[best_model_name]['metrics']['R2']
                        st.metric(
                            "🎯 Model Accuracy",
                            f"{model_confidence:.1%}",
                            "R² Score"
                        )
                    
                    # Show vehicle contributions
                    if contributions:
                        st.markdown("### 📊 Vehicle Type Contributions to Accident Rate")
                        contrib_data = []
                        for vehicle, data in contributions.items():
                            contrib_data.append({
                                'Vehicle Type': vehicle,
                                'Share in Mix': f"{data.get('share', 0):.1%}",
                                'Risk Factor': f"{data.get('risk_factor', 1.0):.1f}x",
                                'Contribution': f"{data.get('relative_contribution', 0):.1%}",
                                'Impact Level': 'High' if data.get('relative_contribution', 0) > 0.2 else 'Medium' if data.get('relative_contribution', 0) > 0.1 else 'Low'
                            })
                        
                        contrib_df = pd.DataFrame(contrib_data)
                        contrib_df = contrib_df.sort_values('Contribution', ascending=False)
                        st.dataframe(contrib_df, hide_index=True)
                        
                        # Create contribution chart
                        fig = px.bar(
                            contrib_df.head(6),  # Top 6 contributors
                            x='Vehicle Type',
                            y=[float(x.strip('%'))/100 for x in contrib_df.head(6)['Contribution']],
                            title='Vehicle Type Contributions to Accident Rate',
                            labels={'y': 'Contribution to Accident Rate', 'x': 'Vehicle Type'},
                            color=[float(x.strip('%'))/100 for x in contrib_df.head(6)['Contribution']],
                            color_continuous_scale='Reds'
                        )
                        fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Provide insights
                    st.markdown("### 🔍 Analysis")
                    
                    # Risk assessment based on vehicle mix
                    normalized_shares = [s/total_share for s in [motor_car_share, motorcycle_share, three_wheeler_share, lorry_share, bus_share, cycle_share, other_share]]
                    motorcycle_norm = normalized_shares[1]  # motorcycle share
                    heavy_vehicle_norm = normalized_shares[3] + normalized_shares[4]  # lorry + bus
                    
                    if prediction > 25:
                        st.error(f"🚨 **High accident rate** ({prediction:.1f} per 1000 vehicles). Urgent safety interventions needed.")
                    elif prediction > 15:
                        st.warning(f"⚠️ **Moderate accident rate** ({prediction:.1f} per 1000 vehicles). Consider targeted safety measures.")
                    else:
                        st.success(f"✅ **Low accident rate** ({prediction:.1f} per 1000 vehicles). Relatively safe traffic environment.")
                    
                    if motorcycle_norm > 0.4:
                        st.warning(f"⚠️ **High motorcycle share** ({motorcycle_norm:.1%}). This significantly increases accident risk due to motorcycle vulnerability.")
                    elif motorcycle_norm > 0.25:
                        st.info(f"ℹ️ **Moderate motorcycle share** ({motorcycle_norm:.1%}). Consider motorcycle safety measures.")
                    
                    if heavy_vehicle_norm > 0.15:
                        st.warning(f"🚛 **High heavy vehicle share** ({heavy_vehicle_norm:.1%}). Focus on heavy vehicle safety and traffic management.")
                    
                    # Show top risk contributor
                    if contributions:
                        top_contributor = max(contributions.items(), key=lambda x: x[1].get('contribution', 0))
                        st.info(f"🎯 **Primary risk factor:** {top_contributor[0]} contributes {top_contributor[1].get('relative_contribution', 0):.1%} to the accident rate.")

    # Model information footer
    st.markdown("---")
    st.markdown("### 🔬 About This Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", best_model_name)
    with col2:
        st.metric("Accuracy", f"{results[best_model_name]['metrics']['R2']:.1%}")
    with col3:
        st.metric("Training Data", "25 provinces")
    
    st.caption("""
    **How it works:** This model uses machine learning trained on Sri Lankan road accident data from 25 provinces. 
    It takes vehicle mix proportions (shares) as input and predicts accident rates (per 1000 vehicles) by analyzing 
    patterns between different vehicle types. The model shows which vehicle types contribute most to accident risk,
    considering factors like traffic interactions, infrastructure, and regional characteristics.
    """)
    
    # Add comparison with real provinces
    with st.expander("📊 Compare with Real Provinces", expanded=False):
        st.markdown("**See how your prediction compares to actual provinces:**")
        
        # Show some real examples
        examples = df[['Location', 'Total_Accidents']].sort_values('Total_Accidents').reset_index(drop=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Lowest Accident Provinces:**")
            for i in range(min(5, len(examples))):
                st.markdown(f"• {examples.iloc[i]['Location']}: {examples.iloc[i]['Total_Accidents']:,} accidents")
        
        with col2:
            st.markdown("**Highest Accident Provinces:**")
            for i in range(max(0, len(examples)-5), len(examples)):
                st.markdown(f"• {examples.iloc[i]['Location']}: {examples.iloc[i]['Total_Accidents']:,} accidents")
        
        avg_accidents = df['Total_Accidents'].mean()
        st.info(f"**Average across all provinces:** {avg_accidents:,.0f} accidents")

if __name__ == "__main__":
    main() 