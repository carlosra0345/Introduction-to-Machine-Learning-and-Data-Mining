import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from regression_dataset import X, targets

def two_fold_cross_validation_ann(attribute_matrix, target_vector, hidden_units_range, K=10):
    outer_kf = KFold(n_splits=K, shuffle=True, random_state=42)
    outer_errors = []

    for train_index, test_index in outer_kf.split(attribute_matrix):
        X_outer_train, X_outer_test = attribute_matrix[train_index], attribute_matrix[test_index]
        y_outer_train, y_outer_test = target_vector[train_index], target_vector[test_index]

        inner_kf = KFold(n_splits=K, shuffle=True, random_state=42)
        best_hidden_units = None
        best_inner_error = float('inf')

        for hidden_units in hidden_units_range:
            fold_errors = []

            for inner_train_index, inner_val_index in inner_kf.split(X_outer_train):
                X_inner_train, X_inner_val = X_outer_train[inner_train_index], X_outer_train[inner_val_index]
                y_inner_train, y_inner_val = y_outer_train[inner_train_index], y_outer_train[inner_val_index]

                model = MLPRegressor(hidden_layer_sizes=(hidden_units,), max_iter=1000, random_state=42)
                model.fit(X_inner_train, y_inner_train.ravel())
                y_inner_pred = model.predict(X_inner_val)
                fold_error = mean_squared_error(y_inner_val, y_inner_pred)
                fold_errors.append(fold_error)

            avg_inner_error = np.mean(fold_errors)

            if avg_inner_error < best_inner_error:
                best_inner_error = avg_inner_error
                best_hidden_units = hidden_units

        # Train the best model on the full outer training set
        best_model = MLPRegressor(hidden_layer_sizes=(best_hidden_units,), max_iter=1000, random_state=42)
        best_model.fit(X_outer_train, y_outer_train.ravel())
        y_outer_pred = best_model.predict(X_outer_test)
        outer_error = mean_squared_error(y_outer_test, y_outer_pred)
        outer_errors.append(outer_error)

    generalization_error = np.mean(outer_errors)
    print(f'ANN Model Generalization Error: {generalization_error}')
    
if __name__ == '__main__':
    X_scaled = X
    y = targets

    hidden_units_range = [1, 5, 10, 20, 50]  # Example range for the number of hidden units
    K = 10  # Number of folds for two-level cross-validation

    print('Performing Two-Level Cross Validation for ANN Model...')
    two_fold_cross_validation_ann(X_scaled, y, hidden_units_range, K)
    print('Complete!')
