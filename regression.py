import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from utils import load_data, preprocess_data, split_data, evaluate_model

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_regression(X_train, y_train):
    """Train Ridge Regression model."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest Regressor."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def compare_models():
    """Compare multiple regression models."""
    print("Loading and preprocessing data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Define models
    models = {
        'Linear Regression': train_linear_regression(X_train, y_train),
        'Ridge Regression': train_ridge_regression(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train)
    }
    
    # Evaluate models
    results = []
    print("\nModel Performance Comparison:")
    print("=" * 50)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MSE': metrics['MSE'],
            'R2': metrics['R2']
        })
        
        print(f"{name}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R²:  {metrics['R2']:.4f}")
        print()
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting MLOps Housing Regression Analysis...")
    print("=" * 60)
    
    # Run model comparison
    results_df = compare_models()
    
    # Save results
    results_df.to_csv('basic_model_results.csv', index=False)
    print("Results saved to 'basic_model_results.csv'")
