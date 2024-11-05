import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from regression_dataset import X, targets

def baseline_model(train_target, test_size):
    mean_value = np.mean(train_target)
    return np.full(test_size, mean_value)

def two_fold_cross_validation_baseline(attribute_matrix, target_vector, K=10):
    outer_kf = KFold(n_splits=K, shuffle=True, random_state=42)
    outer_errors = []

    for train_index, test_index in outer_kf.split(attribute_matrix):
        _, X_outer_test = attribute_matrix[train_index], attribute_matrix[test_index]
        y_outer_train, y_outer_test = target_vector[train_index], target_vector[test_index]

        y_outer_pred = baseline_model(y_outer_train, len(y_outer_test))
        outer_error = mean_squared_error(y_outer_test, y_outer_pred)
        outer_errors.append(outer_error)

    generalization_error = np.mean(outer_errors)
    return generalization_error

if __name__ == '__main__':
    X_scaled = X
    y = targets
    
    K = 10  # Number of folds for two-level cross-validation

    print('Performing Two-Level Cross Validation for Baseline Model...')
    generalization_error = two_fold_cross_validation_baseline(X_scaled, y, K)
    print(f'Baseline Model Generalization Error: {generalization_error}')
    print('Complete!')