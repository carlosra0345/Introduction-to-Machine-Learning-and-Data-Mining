import matplotlib.pyplot as plt
import pandas as pd
from models import logistic_regression
from dataset import X, targets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# Define a range of lambda values
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
cv_errors = []

# Set up K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def fold_error_for_lambda(lam):
    fold_errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        # Use the existing logistic_regression function to define the model
        logistic_model = logistic_regression(lam)
        
        # Fit the model
        logistic_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = logistic_model.predict(X_test)
        
        # Calculate accuracy and convert to error
        accuracy = accuracy_score(y_test, y_pred)
        fold_errors.append(1 - accuracy)  # Convert accuracy to error
    return fold_errors

# Collect cross-validation errors for each lambda
for lam in lambda_values:
    fold_errors = fold_error_for_lambda(lam)
    cv_errors.append(np.mean(fold_errors))  # Average error across folds for current lambda

# Find the optimal lambda
optimal_lambda = lambda_values[np.argmin(cv_errors)]

# Plotting the results
plt.figure()
plt.plot(lambda_values, cv_errors, marker='o')
plt.xscale('log')
plt.xlabel('Regularization parameter λ')
plt.ylabel('Estimated Generalization Error')
plt.title('Generalization Error vs. λ')
plt.grid(True)
plt.savefig("project2/figures/LR_lam.png")


# Train logistic regression with the optimal lambda
logistic_model = logistic_regression(optimal_lambda)  # Using the best lambda
logistic_model.fit(X, targets)  # Assuming X and targets contain your data and labels

# Handle multiclass coefficients by averaging across classes
if logistic_model.coef_.ndim > 1:
    coefficients = logistic_model.coef_.mean(axis=0)  # Take mean across classes for each feature
else:
    coefficients = logistic_model.coef_.flatten()  # Single-class case

# Display the coefficients of logistic regression
feature_importances = pd.DataFrame({
    'Feature': range(X.shape[1]),
    'Coefficient': coefficients
})

print(feature_importances.sort_values(by='Coefficient', ascending=False))

# Plot for visual inspection
feature_importances.sort_values(by='Coefficient', ascending=False, inplace=True)
feature_importances.plot(kind='bar', x='Feature', y='Coefficient', legend=False)
plt.title("Feature Importance in Logistic Regression")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.savefig("project2/figures/LR_plots.png")