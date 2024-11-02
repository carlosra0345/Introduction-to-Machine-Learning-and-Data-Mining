import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Assuming X is your feature matrix and y is your target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lambda_values = np.logspace(-5, 3, num=10)
errors = []

kf = KFold(n_splits=10)

for lambda_value in lambda_values:
    model = Ridge(alpha=lambda_value)
    fold_errors = []
    
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        fold_error = model.score(X_test, y_test)  
        fold_errors.append(fold_error)
    
    avg_error = np.mean(fold_errors)
    errors.append(avg_error)

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, errors, marker='o')
plt.xscale('log')
plt.xlabel('Regularization Parameter λ')
plt.ylabel('Average Generalization Error')
plt.title('Generalization Error vs. Regularization Parameter λ')
plt.grid()
plt.show()
