def load_data():
    """
    Load Boston housing dataset manually due to scikit-learn deprecation.
    Returns DataFrame with features and target variable.
    """
    import pandas as pd
    import numpy as np
    
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Split data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is our target variable
    
    return df

def preprocess_data(df):
    """Preprocess the dataset for training."""
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Handle missing values if any
    X = X.fillna(X.mean())
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics: MSE and R²."""
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'R2': r2
    }

def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5):
    """Perform hyperparameter tuning using GridSearchCV."""
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_
