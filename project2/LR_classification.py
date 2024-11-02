from dataset import glass_identification, X, targets
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Encode target variable for multiclass
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(targets.reshape(-1, 1))

# Define a range of lambda values
lambda_values = np.arange(0.1, 0.6, 0.1) # From 0.0001 to 10
cv_errors = []

# K-Fold Cross-Validation
kf = KFold(n_splits=10)

def fold_error_for_lambda(lam):
    fold_errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        ridge_model = Ridge(alpha=lam)
        ridge_model.fit(X_train, y_train)
        
        y_pred = ridge_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        fold_errors.append(1 - accuracy)  # Convert accuracy to error
    return fold_errors

model_errors = []
for lam in lambda_values:
    fold_errors = fold_error_for_lambda(lam)
    model_errors = model_errors + fold_errors

    # Average the fold errors
    cv_errors.append(np.mean(fold_errors))

# Plotting the results
plt.figure()
plt.plot(lambda_values, cv_errors, marker='o')
plt.xscale('log')
plt.xlabel('Regularization parameter λ')
plt.ylabel('Estimated Generalization Error')
plt.title('Generalization Error vs. λ')
plt.grid(True)
plt.savefig("project2/figures/LR2.png")
