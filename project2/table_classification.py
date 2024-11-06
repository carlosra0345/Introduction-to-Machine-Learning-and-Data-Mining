import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
from models import ann, logistic_regression, baseline, train_neural_net
from dataset import X, targets
from two_level_CV import two_level_cv_ann, two_level_cv_baseline, two_level_cv_logistic_regression

# Ensure targets are zero-indexed
if np.min(targets) == 1:
    targets = targets - 1  # Convert 1-based to 0-based indexing if necessary

# Define ranges for hyperparameters
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]
  # Lambda values from 0.0001 to 10
hidden_units_range = [1, 25, 50, 75, 100, 200, 500]  # Chosen range for hidden units

K1 = 10
K2 = 10

# Higher-level function to perform two-level cross-validation for each model and compile results
def run_two_level_cross_validation(X, y, K1, K2, lambda_range=lambda_values, hidden_units_range=hidden_units_range):
    results = {
        'outer_fold': [],
        'method': [],
        'selected_param': [],
        'test_error': []
    }
    
    outer_cv = KFold(n_splits=K1, shuffle=True, random_state=42)
    
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        
        inner_cv = KFold(n_splits=K2, shuffle=True, random_state=42)
        
        # Logistic Regression
        best_lambda, _ = two_level_cv_logistic_regression(X_train_val, y_train_val, outer_cv, inner_cv, lambda_range)
        logistic_model = logistic_regression(best_lambda)
        logistic_model.fit(X_train_val, y_train_val)
        y_pred = logistic_model.predict(X_test)
        test_error = 1 - accuracy_score(y_test, y_pred)
        results['outer_fold'].append(outer_fold + 1)
        results['method'].append('Logistic Regression')
        results['selected_param'].append(best_lambda)
        results['test_error'].append(test_error)
        
        # ANN
        best_h, _ = two_level_cv_ann(X_train_val, y_train_val, outer_cv, inner_cv, hidden_units_range)
        ann_model = ann(best_h)
        trained_ann_model, _ = train_neural_net(ann_model, F.cross_entropy, X_train_val, y_train_val)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_pred = trained_ann_model(X_test_tensor)
        y_test_pred = torch.max(y_test_pred, dim=1)[1].data.numpy()
        test_error = np.mean(y_test_pred != y_test)
        results['outer_fold'].append(outer_fold + 1)
        results['method'].append('ANN')
        results['selected_param'].append(best_h)
        results['test_error'].append(test_error)
        
        # Baseline
        baseline_error = two_level_cv_baseline(X_train_val, y_train_val, outer_cv)
        results['outer_fold'].append(outer_fold + 1)
        results['method'].append('Baseline')
        results['selected_param'].append('N/A')
        results['test_error'].append(baseline_error)

    # Convert results to DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df

# Run the two-level cross-validation and print results
results_df = run_two_level_cross_validation(X, targets, K1, K2)
print(results_df)

# Assuming results_df is the DataFrame output from run_two_level_cross_validation
# For reference, results_df has columns: ['outer_fold', 'method', 'selected_param', 'test_error']

def find_optimal_params(results_df):
    # Filter for each model type
    logistic_results = results_df[results_df['method'] == 'Logistic Regression']
    ann_results = results_df[results_df['method'] == 'ANN']
    
    # Find the best lambda for logistic regression (smallest test error)
    optimal_lambda_row = logistic_results.loc[logistic_results['test_error'].idxmin()]
    optimal_lambda = optimal_lambda_row['selected_param']
    optimal_lambda_error = optimal_lambda_row['test_error']
    
    # Find the best h (hidden units) for ANN (smallest test error)
    optimal_h_row = ann_results.loc[ann_results['test_error'].idxmin()]
    optimal_h = optimal_h_row['selected_param']
    optimal_h_error = optimal_h_row['test_error']
    
    # Return optimal parameters and their corresponding errors
    return {
        'optimal_lambda': optimal_lambda,
        'optimal_lambda_error': optimal_lambda_error,
        'optimal_h': optimal_h,
        'optimal_h_error': optimal_h_error
    }

# Example usage:
optimal_params = find_optimal_params(results_df)
print(f"Optimal lambda for logistic regression: {optimal_params['optimal_lambda']} with error {optimal_params['optimal_lambda_error']}")
print(f"Optimal h for ANN: {optimal_params['optimal_h']} with error {optimal_params['optimal_h_error']}")

def generate_latex_table(cv_results):
    """
    Generates LaTeX code for a table based on the two-level cross-validation results.

    Parameters:
        cv_results (pd.DataFrame): DataFrame with columns 'outer_fold', 'method', 'selected_param', and 'test_error'.
        
    Returns:
        str: LaTeX code for the table.
    """
    # Initialize the LaTeX table structure
    latex_code = r"""
\begin{table}[h!]
\centering
\begin{tabular}{cccccc}
\hline
\textbf{Outer fold} & \multicolumn{2}{c}{\textbf{Method 2}} & \multicolumn{2}{c}{\textbf{Logistic regression}} & \textbf{Baseline} \\
$i$ & $x^*_i$ & $E^{\text{test}}_i$ & $\lambda^*_i$ & $E^{\text{test}}_i$ & $E^{\text{test}}_i$ \\
\hline
"""

    # Iterate through each outer fold
    unique_folds = cv_results['outer_fold'].unique()
    for fold in unique_folds:
        fold_data = cv_results[cv_results['outer_fold'] == fold]
        
        # Extract data for each model
        ann_row = fold_data[fold_data['method'] == 'ANN'].iloc[0]
        log_reg_row = fold_data[fold_data['method'] == 'Logistic Regression'].iloc[0]
        baseline_row = fold_data[fold_data['method'] == 'Baseline'].iloc[0]
        
        # Add row data to the table with full precision for all values
        latex_code += (
                    f"{fold} & "
                    f"{ann_row['selected_param']} & {ann_row['test_error']:.4f} & "
                    f"{log_reg_row['selected_param']:.4f} & {log_reg_row['test_error']:.4f} & "
                    f"{baseline_row['test_error']:.4f} \\\\\n"
                )
    # Close the table structure
    latex_code += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex_code

# Generate LaTeX code
latex_table_code = generate_latex_table(results_df)
print(latex_table_code)
