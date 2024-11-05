import numpy as np
from ann_regression import two_fold_cross_validation_ann
from baseline_regression import two_fold_cross_validation_baseline
from linear_regression import two_fold_cross_validation
from regression_dataset import X, targets

# Define parameters
hidden_units_range = [1, 5, 10, 20, 50]
lambdas = np.logspace(-5, 3, 10)
K = 10  # Number of folds for cross-validation

# Data
X_scaled = X
y = targets

# Run ANN model cross-validation
print('Performing Two-Level Cross Validation for ANN Model...')
ann_generalization_error = two_fold_cross_validation_ann(X_scaled, y, hidden_units_range, K)
print('Complete!\n')

# Run Baseline model cross-validation
print('Performing Two-Level Cross Validation for Baseline Model...')
baseline_generalization_error = two_fold_cross_validation_baseline(X_scaled, y, K)
print('Complete!\n')

# Run Regularized Linear Regression model cross-validation
print('Performing Two-Level Cross Validation for Regularized Linear Regression Model...')
ridge_generalization_error = two_fold_cross_validation(X_scaled, y, lambdas, K)
print('Complete!\n')

# Print comparative table
print(f"{'Model':<30}{'Generalization Error':<30}")
print('-' * 60)
print(f"{'ANN Model':<30}{ann_generalization_error:<30}")
print(f"{'Baseline Model':<30}{baseline_generalization_error:<30}")
print(f"{'Regularized Linear Regression':<30}{ridge_generalization_error:<30}")
