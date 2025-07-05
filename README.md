# MLOps Assignment - Housing Regression

**GitHub Repository:** https://github.com/yourusername/HousingRegression

## Project Overview

This project implements a complete MLOps workflow for predicting house prices using the Boston Housing dataset. The implementation includes automated CI/CD pipeline, model comparison, and hyperparameter tuning following MLOps best practices.

## Repository Structure

HousingRegression/
├── .github/workflows/
│ └── ci.yml # GitHub Actions CI/CD pipeline
├── utils.py # Utility functions for data processing
├── regression.py # Main regression models and analysis
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── basic_model_results.csv # Basic model performance results
└── tuned_model_results.csv # Hyperparameter tuned results


## Models Implemented

### 1. Linear Regression
- **Purpose:** Baseline linear model for comparison
- **Hyperparameters Tuned:**
  - `fit_intercept`: [True, False]
  - `positive`: [True, False]
  - `copy_X`: [True, False]

### 2. Ridge Regression  
- **Purpose:** L2 regularization to prevent overfitting
- **Hyperparameters Tuned:**
  - `alpha`: [0.1, 1.0, 10.0, 100.0]
  - `fit_intercept`: [True, False]
  - `solver`: ['auto', 'svd', 'cholesky', 'lsqr']

### 3. Random Forest Regressor
- **Purpose:** Ensemble method for robust predictions
- **Hyperparameters Tuned:**
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [None, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10]

## Performance Metrics

Models are evaluated using:
- **Mean Squared Error (MSE):** Lower is better
- **R² Score:** Higher is better (max = 1.0)

## Installation and Usage

### Prerequisites
- Python 3.8+
- Git
- GitHub account

### Setup Instructions

1. **Clone the repository:**
git clone https://github.com/yourusername/HousingRegression.git
cd HousingRegression



2. **Create conda environment:**
conda create -n mlops-housing python=3.8
conda activate mlops-housing



3. **Install dependencies:**
pip install -r requirements.txt



4. **Run the analysis:**
python regression.py


## CI/CD Pipeline

The project includes automated GitHub Actions workflow that:
- Runs on push to `main`, `reg-branch`, and `hyper-branch`
- Installs dependencies automatically
- Executes regression analysis
- Generates performance reports
- Validates code functionality

## Branch Structure

- **`main`:** Production-ready code with complete functionality
- **`reg-branch`:** Basic regression models implementation
- **`hyper-branch`:** Hyperparameter tuning features

## Dataset Information

The Boston Housing dataset contains 506 samples with 13 features:
- **Target Variable:** MEDV (Median home value)
- **Features:** CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT

**Note:** Dataset is loaded from original CMU source due to scikit-learn deprecation.

## Assignment Requirements Fulfilled

- [x] Minimum 3 regression models implemented
- [x] Performance comparison using MSE and R²
- [x] Hyperparameter tuning with minimum 3 parameters per model
- [x] GitHub Actions CI/CD pipeline
- [x] Proper branch management (main, reg-branch, hyper-branch)
- [x] Modular code structure with utils.py
- [x] Complete documentation

## Author

**Your Name** - Aditya Sharma G24AI1098  
**Email:** G24AI1098@iitj,ac,in  
