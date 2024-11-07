import numpy as np
from ann_regression import two_fold_cross_validation_ann
from baseline_regression import two_fold_cross_validation_baseline
from linear_regression import two_fold_cross_validation
from regression_dataset import X, targets
import csv

# Define parameters
hidden_units_range = [1, 5, 10, 20, 50]
lambdas = np.logspace(-5, 3, 10)
K = 10  # Number of folds for cross-validation

# Data
X_scaled = X
y = targets

# Run ANN model cross-validation
print('Performing Two-Level Cross Validation for ANN Model...')
ann_generalization_error, ann_generalization_errors, ann_hidden_units = two_fold_cross_validation_ann(X_scaled, y, hidden_units_range, K)
print('Complete!\n')

# Run Baseline model cross-validation
print('Performing Two-Level Cross Validation for Baseline Model...')
baseline_generalization_error, baseline_generalization_errors = two_fold_cross_validation_baseline(X_scaled, y, K)
print('Complete!\n')

# Run Regularized Linear Regression model cross-validation
print('Performing Two-Level Cross Validation for Regularized Linear Regression Model...')
ridge_generalization_error, ridge_generalization_errors, ridge_regularization_strengths = two_fold_cross_validation(X_scaled, y, lambdas, K)
print('Complete!\n')

# save generalization error table
with open('figures/generalization_error.csv', mode='w', newline='') as file:
    headers = ['Model', 'Generalization Error']
    writer = csv.writer(file)
    writer.writerow(headers)
    
    # Write the data rows to the CSV file
    data = [
        ['ANN Model', ann_generalization_error],
        ['Baseline Model', baseline_generalization_error],
        ['Regularized Linear Regression', ridge_generalization_error]
    ]
    writer.writerows(data)
    
# save main report table
with open('figures/report_table.csv', mode='w', newline='') as file:
    parent = ['Outer Fold', '', 'ANN', '', '', 'Linear Regression', '', '', 'Baseline']
    children = ['i', '', 'h_i^*', 'E_i^test', '', 'lambda_i^*', 'E_i^test', '', 'E_i^test']
    writer = csv.writer(file)
    writer.writerow(parent)
    writer.writerow(children)
    
    data = []
    for fold_num in range(len(ridge_generalization_errors)):
        data.append([
            fold_num + 1, 
            '',
            ann_hidden_units[fold_num],
            ann_generalization_errors[fold_num],
            '',
            ridge_regularization_strengths[fold_num],
            ridge_generalization_errors[fold_num],
            '',
            baseline_generalization_errors[fold_num]
        ])
    writer.writerows(data)
