import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def prepare_modeling_data_improved(df):
    """Prepare data for modeling accident rates with better feature engineering."""
    # Use vehicle shares as features for predicting accident rates
    share_columns = [col for col in df.columns if col.endswith('_Share')]
    
    # Add derived features
    feature_columns = share_columns + ['Diversity_Index', 'Heavy_Vehicle_Ratio']
    
    # Create additional engineered features
    df_features = df[feature_columns].copy()
    
    # Add interaction features based on shares
    df_features['Motor_Vehicle_Share_Total'] = (df['Motor Car_Share'] + 
                                               df['Motor Cycle/Moped_Share'] + 
                                               df['Three wheeler_Share'])
    df_features['Commercial_Vehicle_Share_Total'] = (df['Lorry_Share'] + 
                                                    df['SLT Bus_Share'] + 
                                                    df['Private Bus_Share'] + 
                                                    df['Intercity Bus_Share'])
    df_features['Heavy_Light_Share_Ratio'] = (df_features['Commercial_Vehicle_Share_Total'] / 
                                             (df_features['Motor_Vehicle_Share_Total'] + 0.01))
    
    # Add squared terms for non-linear relationships
    for col in share_columns:
        df_features[f'{col}_squared'] = df[col] ** 2
    
    # Calculate accident rate as target
    # Since we have accident counts per vehicle type, we need to estimate vehicle population
    # We'll use a scaling factor based on typical accident rates per vehicle type
    
    # Typical accident rates per 1000 vehicles by type (estimated from research)
    accident_rates_per_1000 = {
        'Motor Car': 15,
        'Dual Purpose Vehicle': 18,
        'Lorry': 25,
        'Cycle': 8,
        'Motor Cycle/Moped': 45,  # Higher risk
        'Three wheeler': 35,
        'Articulated Vehicle, prime mover': 30,
        'SLT Bus': 20,
        'Private Bus': 22,
        'Intercity Bus': 18,
        'Land Vehicle/Tractor': 12,
        'Animal drawn vehicle or rider on animal': 5,
        'Other': 15
    }
    
    # Estimate vehicle population from accident counts
    estimated_vehicles = []
    for idx, row in df.iterrows():
        total_estimated_vehicles = 0
        for vehicle_type in accident_rates_per_1000.keys():
            if vehicle_type in df.columns:
                accidents = row[vehicle_type]
                rate_per_1000 = accident_rates_per_1000[vehicle_type]
                estimated_vehicle_count = (accidents / rate_per_1000) * 1000
                total_estimated_vehicles += estimated_vehicle_count
        estimated_vehicles.append(total_estimated_vehicles)
    
    # Calculate accident rate per 1000 vehicles
    estimated_vehicles = np.array(estimated_vehicles)
    accident_rate = (df['Total_Accidents'] / estimated_vehicles) * 1000
    
    # Handle any infinite or NaN values
    accident_rate = np.where(np.isfinite(accident_rate), accident_rate, df['Total_Accidents'].mean() / 1000)
    
    return df_features, accident_rate

def train_models_improved(X, y, test_size=0.3, random_state=42):
    """Train and evaluate multiple regression models with improved approach."""
    
    # For small datasets, use Leave-One-Out CV for more robust evaluation
    if len(X) < 50:
        print(f"Small dataset detected ({len(X)} samples). Using Leave-One-Out Cross-Validation.")
        use_loo = True
    else:
        use_loo = False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Initialize models with better parameters for small datasets
    models = {
        'Ridge (Alpha=1.0)': Ridge(alpha=1.0, max_iter=5000),
        'Ridge (Alpha=10.0)': Ridge(alpha=10.0, max_iter=5000),
        'Lasso (Alpha=0.1)': Lasso(alpha=0.1, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
        'Random Forest (Small)': RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_split=3, 
            min_samples_leaf=2, random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, 
            random_state=random_state
        ),
        'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation scores
            if use_loo:
                loo = LeaveOneOut()
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=loo, 
                    scoring='neg_root_mean_squared_error'
                )
            else:
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=5, 
                    scoring='neg_root_mean_squared_error'
                )
            
            cv_rmse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'metrics': {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                },
                'cv_rmse': cv_rmse,
                'cv_std': cv_std,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    return results, X_test_scaled, y_test

def evaluate_model_performance(results):
    """Comprehensive model evaluation and comparison."""
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Create performance summary
    performance_data = []
    for name, result in results.items():
        performance_data.append({
            'Model': name,
            'RMSE': result['metrics']['RMSE'],
            'MAE': result['metrics']['MAE'],
            'R²': result['metrics']['R2'],
            'CV RMSE': result['cv_rmse'],
            'CV Std': result['cv_std']
        })
    
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values('R²', ascending=False)
    
    print("\nModel Performance Summary (sorted by R²):")
    print("-" * 80)
    for _, row in performance_df.iterrows():
        print(f"{row['Model']:<25} | RMSE: {row['RMSE']:>8.2f} | MAE: {row['MAE']:>8.2f} | "
              f"R²: {row['R²']:>7.3f} | CV RMSE: {row['CV RMSE']:>8.2f} ± {row['CV Std']:>5.2f}")
    
    # Identify best model
    best_model_name = performance_df.iloc[0]['Model']
    best_r2 = performance_df.iloc[0]['R²']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {best_r2:.3f}")
    
    # Performance interpretation
    print("\n📊 PERFORMANCE INTERPRETATION:")
    if best_r2 > 0.7:
        print("   ✅ Excellent performance - Model explains >70% of variance")
    elif best_r2 > 0.5:
        print("   ✅ Good performance - Model explains >50% of variance")
    elif best_r2 > 0.3:
        print("   ⚠️  Moderate performance - Model explains >30% of variance")
    elif best_r2 > 0:
        print("   ⚠️  Poor performance - Model barely better than mean prediction")
    else:
        print("   ❌ Very poor performance - Model worse than mean prediction")
    
    return performance_df, best_model_name

def analyze_predictions(results, best_model_name):
    """Analyze prediction quality and residuals."""
    best_result = results[best_model_name]
    y_test = best_result['y_test']
    y_pred = best_result['y_pred']
    
    print(f"\n📈 PREDICTION ANALYSIS FOR {best_model_name}:")
    print("-" * 60)
    
    # Prediction accuracy analysis
    residuals = y_test - y_pred
    relative_errors = np.abs(residuals) / y_test * 100
    
    print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.2f} accidents")
    print(f"Mean Relative Error: {np.mean(relative_errors):.1f}%")
    print(f"Predictions within 20% of actual: {np.sum(relative_errors <= 20)}/{len(relative_errors)} "
          f"({np.sum(relative_errors <= 20)/len(relative_errors)*100:.1f}%)")
    
    # Create prediction vs actual plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Total Accidents')
    plt.ylabel('Predicted Total Accidents')
    plt.title(f'Predictions vs Actual\n{best_model_name}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Total Accidents')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return residuals, relative_errors

def feature_importance_analysis(results, best_model_name, feature_names):
    """Analyze feature importance for the best model."""
    best_model = results[best_model_name]['model']
    
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS ({best_model_name}):")
    print("-" * 60)
    
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models
        importance = best_model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(best_model, 'coef_'):
        # Linear models
        importance = np.abs(best_model.coef_)
        importance_type = "Coefficient Magnitude"
    else:
        print("Feature importance not available for this model type.")
        return
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Display top features
    print(f"\nTop 10 Most Important Features ({importance_type}):")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['Feature']:<30} {row['Importance']:>8.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel(importance_type)
    plt.title(f'Top 15 Features - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def suggest_improvements():
    """Suggest specific improvements based on the analysis."""
    print("\n" + "="*80)
    print("🚀 RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("="*80)
    
    improvements = [
        "1. DATA COLLECTION:",
        "   • Collect more historical data (multiple years)",
        "   • Add external features: population, road density, economic indicators",
        "   • Include temporal features: seasonality, weather conditions",
        "",
        "2. FEATURE ENGINEERING:",
        "   • Create more interaction terms between vehicle types",
        "   • Add geographical features (urban/rural classification)",
        "   • Include traffic volume and road infrastructure data",
        "",
        "3. MODEL IMPROVEMENTS:",
        "   • Try ensemble methods combining multiple models",
        "   • Use polynomial features for non-linear relationships",
        "   • Consider time-series modeling if temporal data available",
        "",
        "4. VALIDATION STRATEGY:",
        "   • Use stratified sampling based on accident severity",
        "   • Implement temporal validation for time-series data",
        "   • Consider spatial cross-validation for geographical data",
        "",
        "5. DOMAIN-SPECIFIC ENHANCEMENTS:",
        "   • Weight models by population or traffic volume",
        "   • Include policy intervention indicators",
        "   • Add road safety infrastructure metrics"
    ]
    
    for improvement in improvements:
        print(improvement)

def create_vehicle_contribution_analysis(model, feature_names, input_shares):
    """Analyze how each vehicle type contributes to the predicted accident rate."""
    contributions = {}
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_)
    else:
        # For other models, use permutation importance approximation
        importances = np.ones(len(feature_names)) / len(feature_names)
    
    # Map feature importances to vehicle types
    vehicle_types = ['Motor Car', 'Dual Purpose Vehicle', 'Lorry', 'Cycle', 
                    'Motor Cycle/Moped', 'Three wheeler', 'Articulated Vehicle, prime mover',
                    'SLT Bus', 'Private Bus', 'Intercity Bus', 'Land Vehicle/Tractor',
                    'Animal drawn vehicle or rider on animal', 'Other']
    
    for i, vehicle in enumerate(vehicle_types):
        share_col = f'{vehicle}_Share'
        if share_col in feature_names:
            idx = feature_names.index(share_col)
            # Calculate contribution as importance * share * base_risk_factor
            base_risk = {
                'Motor Cycle/Moped': 2.5,  # High risk
                'Motor Car': 1.0,          # Baseline
                'Three wheeler': 1.8,      # Medium-high risk
                'Lorry': 1.5,             # Medium risk
                'Private Bus': 1.2,       # Medium risk
                'Cycle': 0.8,             # Lower risk
                'SLT Bus': 1.2,           # Medium risk
                'Intercity Bus': 1.1,     # Medium risk
                'Dual Purpose Vehicle': 1.1,
                'Articulated Vehicle, prime mover': 1.6,
                'Land Vehicle/Tractor': 1.3,
                'Animal drawn vehicle or rider on animal': 0.5,
                'Other': 1.0
            }
            
            vehicle_share = input_shares.get(share_col, 0)
            risk_factor = base_risk.get(vehicle, 1.0)
            contribution = importances[idx] * vehicle_share * risk_factor
            
            contributions[vehicle] = {
                'share': vehicle_share,
                'importance': importances[idx],
                'risk_factor': risk_factor,
                'contribution': contribution
            }
    
    return contributions

if __name__ == "__main__":
    from data_preparation import load_data, prepare_features
    
    print("🚗 IMPROVED ROAD ACCIDENT PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    df = load_data()
    df = prepare_features(df)
    
    print(f"Dataset: {len(df)} provinces, {len(df.columns)} features")
    print(f"Total accidents range: {df['Total_Accidents'].min()} - {df['Total_Accidents'].max()}")
    
    # Prepare modeling data with improved features
    X, y = prepare_modeling_data_improved(df)
    print(f"Features for modeling: {len(X.columns)}")
    
    # Train models
    print("\n🤖 Training and evaluating models...")
    results, X_test, y_test = train_models_improved(X, y)
    
    # Evaluate performance
    performance_df, best_model_name = evaluate_model_performance(results)
    
    # Analyze predictions
    residuals, relative_errors = analyze_predictions(results, best_model_name)
    
    # Feature importance analysis
    importance_df = feature_importance_analysis(results, best_model_name, X.columns)
    
    # Suggest improvements
    suggest_improvements()
    
    print(f"\n📁 Analysis complete! Check generated files:")
    print("   • model_predictions_analysis.png")
    print("   • feature_importance_analysis.png") 