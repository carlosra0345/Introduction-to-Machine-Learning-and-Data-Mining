import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
from models import ann, logistic_regression, baseline, train_neural_net
from dataset import X, targets

# Define two-level cross-validation for logistic regression
def two_level_cv_logistic_regression(X, y, outer_cv, inner_cv, lambda_range):
    best_lambda, best_test_error = None, float('inf')
    for lmbda in lambda_range:
        fold_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X):
            X_inner_train, X_inner_val = X[inner_train_idx], X[inner_val_idx]
            y_inner_train, y_inner_val = y[inner_train_idx], y[inner_val_idx]
            
            # Train logistic regression with regularization parameter lambda
            logistic_model = logistic_regression(lmbda)
            logistic_model.fit(X_inner_train, y_inner_train)
            y_pred = logistic_model.predict(X_inner_val)
            error = 1 - accuracy_score(y_inner_val, y_pred)
            fold_errors.append(error)
            
        mean_error = np.mean(fold_errors)
        if mean_error < best_test_error:
            best_test_error = mean_error
            best_lambda = lmbda
    return best_lambda, best_test_error

# Define two-level cross-validation for ANN
def two_level_cv_ann(X, y, outer_cv, inner_cv, hidden_units_range):
    best_h, best_test_error = None, float('inf')
    for h in hidden_units_range:
        fold_errors = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X):
            X_inner_train, X_inner_val = X[inner_train_idx], X[inner_val_idx]
            y_inner_train, y_inner_val = y[inner_train_idx], y[inner_val_idx]
            
            # Train ANN with specified number of hidden units
            model = ann(h)
            trained_model, _ = train_neural_net(model, F.cross_entropy, X_inner_train, y_inner_train)
            
            X_inner_val_tensor = torch.tensor(X_inner_val, dtype=torch.float32)
            y_val_pred = trained_model(X_inner_val_tensor)
            y_val_pred = torch.max(y_val_pred, dim=1)[1].data.numpy()
            error = np.mean(y_val_pred != y_inner_val)
            fold_errors.append(error)
        
        mean_error = np.mean(fold_errors)
        if mean_error < best_test_error:
            best_test_error = mean_error
            best_h = h
    return best_h, best_test_error

# Define two-level cross-validation for Baseline
def two_level_cv_baseline(X, y, outer_cv):
    baseline_model = baseline()
    baseline_model.fit(X, y)
    y_pred = baseline_model.predict(X)
    test_error = 1 - accuracy_score(y, y_pred)
    return test_error