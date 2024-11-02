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

    # Plot lambda vs generalization error
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, generalization_errors, marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('Lambda (Regularization Parameter)')
    plt.ylabel('Average Generalization Error (MSE)')
    plt.title('Generalization Error vs. Lambda')
    plt.grid(True)
    plt.savefig('project2/figures/linear_regression_lambda_vs_error.png')

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
    K_Fold_cross_validation(X_scaled, y, lambdas, 10)
    print('Complete!')
elif user_input == 2:
    print('Performing Two-Level Cross Validation...')
else:
    print('Error: Invalid user input')