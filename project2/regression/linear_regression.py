import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from regression_dataset import X, targets

def K_Fold_cross_validation(attribute_matrix, target_vector, lambdas, K=10):
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    generalization_errors = []  # Store generalization error for each lambda

    # Loop over each lambda value
    for lam in lambdas:
        model = Ridge(alpha=lam)
        fold_errors = []  # Store errors for each fold
        
        # Perform K-Fold cross-validation
        for train_index, test_index in kf.split(attribute_matrix):
            X_train, X_test = attribute_matrix[train_index], attribute_matrix[test_index]
            y_train, y_test = target_vector[train_index], target_vector[test_index]
            
            # Train the model on the training set
            model.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = model.predict(X_test)
            
            # Calculate the test error (Mean Squared Error) for this fold
            fold_error = mean_squared_error(y_test, y_pred)
            fold_errors.append(fold_error)
        
        # Calculate the average generalization error for this lambda
        generalization_error = np.mean(fold_errors)
        generalization_errors.append(generalization_error)

    min_err = float('inf')
    optimal_lam = -100000
    for i, error in enumerate(generalization_errors):
        if error < min_err:
            optimal_lam = lambdas[i]
            min_err = error
                
    # Plot lambda vs generalization error
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, generalization_errors, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('Average Generalization Error (MSE)')
    plt.title('Generalization Error vs. Lambda')
    plt.grid(True)
    plt.savefig('figures/linear_regression_lambda_vs_error.png')
    return optimal_lam
    
def two_fold_cross_validation(attribute_matrix, target_vector, lambdas, K):
    outer_kf = KFold(n_splits=K, shuffle=True, random_state=42)
    outer_errors = []

    for train_index, test_index in outer_kf.split(attribute_matrix):
        X_outer_train, X_outer_test = attribute_matrix[train_index], attribute_matrix[test_index]
        y_outer_train, y_outer_test = target_vector[train_index], target_vector[test_index]

        inner_kf = KFold(n_splits=K, shuffle=True, random_state=42)
        best_lambda = None
        best_inner_error = float('inf')

        for lam in lambdas:
            fold_errors = []

            for inner_train_index, inner_val_index in inner_kf.split(X_outer_train):
                X_inner_train, X_inner_val = X_outer_train[inner_train_index], X_outer_train[inner_val_index]
                y_inner_train, y_inner_val = y_outer_train[inner_train_index], y_outer_train[inner_val_index]

                model = Ridge(alpha=lam)
                model.fit(X_inner_train, y_inner_train)
                y_inner_pred = model.predict(X_inner_val)
                fold_error = mean_squared_error(y_inner_val, y_inner_pred)
                fold_errors.append(fold_error)

            avg_inner_error = np.mean(fold_errors)

            if avg_inner_error < best_inner_error:
                best_inner_error = avg_inner_error
                best_lambda = lam

        # Train on the full outer training set with the best lambda
        best_model = Ridge(alpha=best_lambda)
        best_model.fit(X_outer_train, y_outer_train)
        y_outer_pred = best_model.predict(X_outer_test)
        outer_error = mean_squared_error(y_outer_test, y_outer_pred)
        outer_errors.append(outer_error)

    generalization_error = np.mean(outer_errors)
    return generalization_error

if __name__ == '__main__':
    X_scaled = X
    y = targets

    # Define range of lambda values
    lambdas = np.logspace(-5, 3, 10)  # From 10^-5 to 10^23 with finer granularity

    K = 10  # Number of folds for K-Fold cross-validation

    print('1: K-Fold Cross Validation')
    print('2: Two-Level Cross Validation')
    user_input = int(input('Enter which cross validation method to use: '))
    print('----------------------------------------------')

    if user_input == 1:
        print('Performing K-Fold Cross Validation...')
        generalization_error = K_Fold_cross_validation(X_scaled, y, lambdas, K)
        print(f'K-Fold Cross Validation Error: {generalization_error}')
        print('Complete!')
    elif user_input == 2:
        print('Performing Two-Level Cross Validation...')
        generalization_error = two_fold_cross_validation(X_scaled, y, lambdas, K)
        print(f'Two-Level Cross-Validation Generalization Error: {generalization_error}')
        print('Complete!')
    else:
        print('Error: Invalid user input')