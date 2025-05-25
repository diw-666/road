#!/usr/bin/env python3
"""
Test script to verify the prediction functionality works correctly.
"""

import pandas as pd
import numpy as np
from data_preparation import load_data, prepare_features
from predictive_modeling import prepare_modeling_data_improved, train_models_improved

def create_model_input(motor_car, motorcycle, three_wheeler, lorry, bus, cycle, other_vehicles):
    """Create properly formatted input data for the model prediction."""
    
    # Define the exact vehicle columns the model expects
    vehicle_columns = [
        'Motor Car', 'Dual Purpose Vehicle', 'Lorry', 'Cycle', 
        'Motor Cycle/Moped', 'Three wheeler', 'Articulated Vehicle, prime mover',
        'SLT Bus', 'Private Bus', 'Intercity Bus', 'Land Vehicle/Tractor',
        'Animal drawn vehicle or rider on animal', 'Other'
    ]
    
    # Create input data with proper mapping
    input_data = pd.DataFrame([{
        'Motor Car': motor_car,
        'Motor Cycle/Moped': motorcycle,
        'Three wheeler': three_wheeler,
        'Lorry': lorry,
        'Private Bus': bus,
        'Cycle': cycle,
        'Dual Purpose Vehicle': other_vehicles * 0.4,
        'SLT Bus': other_vehicles * 0.1,
        'Intercity Bus': other_vehicles * 0.1,
        'Articulated Vehicle, prime mover': other_vehicles * 0.1,
        'Land Vehicle/Tractor': other_vehicles * 0.2,
        'Animal drawn vehicle or rider on animal': other_vehicles * 0.05,
        'Other': other_vehicles * 0.05
    }])
    
    # Add derived features that the model expects
    total_accidents = input_data.sum(axis=1).iloc[0]
    input_data['Diversity_Index'] = 0.8  # Reasonable default
    input_data['Heavy_Vehicle_Ratio'] = (lorry + bus) / total_accidents if total_accidents > 0 else 0
    
    # Add engineered features in the exact order the model expects
    input_data['Motor_Vehicle_Total'] = input_data['Motor Car'] + input_data['Motor Cycle/Moped'] + input_data['Three wheeler']
    input_data['Commercial_Vehicle_Total'] = input_data['Lorry'] + input_data['SLT Bus'] + input_data['Private Bus'] + input_data['Intercity Bus']
    input_data['Heavy_Light_Ratio'] = input_data['Commercial_Vehicle_Total'] / (input_data['Motor_Vehicle_Total'] + 1)
    
    # Add log features in the exact order the model expects
    for col in vehicle_columns:
        input_data[f'{col}_log'] = np.log1p(input_data[col])
    
    return input_data

def test_prediction():
    """Test the prediction functionality."""
    print("🧪 Testing Prediction Functionality")
    print("=" * 50)
    
    # Load and prepare data
    print("📊 Loading data...")
    df = load_data()
    df = prepare_features(df)
    
    # Prepare modeling data
    print("🔧 Preparing modeling data...")
    X, y = prepare_modeling_data_improved(df)
    results, X_test, y_test = train_models_improved(X, y)
    
    # Get best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
    best_model = results[best_model_name]['model']
    scaler = results[best_model_name]['scaler']
    
    print(f"🏆 Best model: {best_model_name}")
    print(f"📈 R² Score: {results[best_model_name]['metrics']['R2']:.3f}")
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Small Rural Province",
            "motor_car": 10,
            "motorcycle": 70,
            "three_wheeler": 25,
            "lorry": 30,
            "bus": 8,
            "cycle": 20,
            "other_vehicles": 15
        },
        {
            "name": "Medium Province",
            "motor_car": 120,
            "motorcycle": 380,
            "three_wheeler": 300,
            "lorry": 240,
            "bus": 110,
            "cycle": 65,
            "other_vehicles": 80
        },
        {
            "name": "Large Urban Province",
            "motor_car": 700,
            "motorcycle": 1300,
            "three_wheeler": 850,
            "lorry": 520,
            "bus": 400,
            "cycle": 200,
            "other_vehicles": 180
        }
    ]
    
    print("\n🎯 Testing Prediction Scenarios:")
    print("-" * 50)
    
    for scenario in test_scenarios:
        print(f"\n📍 {scenario['name']}:")
        
        # Create input data
        input_data = create_model_input(
            scenario['motor_car'],
            scenario['motorcycle'],
            scenario['three_wheeler'],
            scenario['lorry'],
            scenario['bus'],
            scenario['cycle'],
            scenario['other_vehicles']
        )
        
        # Scale input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = best_model.predict(input_data_scaled)[0]
        
        # Calculate input sum
        input_sum = sum([
            scenario['motor_car'],
            scenario['motorcycle'],
            scenario['three_wheeler'],
            scenario['lorry'],
            scenario['bus'],
            scenario['cycle'],
            scenario['other_vehicles']
        ])
        
        # Calculate difference
        diff = prediction - input_sum
        diff_pct = (diff / input_sum) * 100
        
        print(f"   Input sum: {input_sum:,.0f}")
        print(f"   Predicted: {prediction:,.0f}")
        print(f"   Difference: {diff:+,.0f} ({diff_pct:+.1f}%)")
        
        # Validate prediction is reasonable
        if 0 < prediction < 50000:  # Reasonable range
            print("   ✅ Prediction looks reasonable")
        else:
            print("   ⚠️  Prediction may be unrealistic")
    
    print("\n✅ Prediction test completed successfully!")

if __name__ == "__main__":
    test_prediction() 