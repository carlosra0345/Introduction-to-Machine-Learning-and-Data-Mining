import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
from models import ann, logistic_regression, baseline, train_neural_net
from dataset import glass_identification, X, targets
from two_level_CV import two_level_cv_ann, two_level_cv_baseline, two_level_cv_logistic_regression
from table_classification import results_df

# Ensure targets are zero-indexed
if np.min(targets) == 1:
    targets = targets - 1  # Convert 1-based to 0-based indexing if necessary

# Define function to perform the correlated t-test
def correlated_t_test(r_j, J, alpha=0.05):
    r_mean = np.mean(r_j)
    s_squared = np.var(r_j, ddof=1)
    rho = 1 / J
    sigma_tilde_squared = s_squared * (1 / J + rho / (1 - rho))
    nu = J - 1
    t_statistic = r_mean / np.sqrt(sigma_tilde_squared)
    p_value = 2 * t.cdf(-abs(t_statistic), df=nu)
    confidence_interval = (
        r_mean - t.ppf(1 - alpha / 2, nu) * np.sqrt(sigma_tilde_squared),
        r_mean + t.ppf(1 - alpha / 2, nu) * np.sqrt(sigma_tilde_squared)
    )
    return r_mean, confidence_interval, p_value

# Function to get ANN predictions
def get_ann_predictions(model_fn, X_train, y_train, X_test):
    ann_model, _ = train_neural_net(model_fn, F.cross_entropy, X_train, y_train)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = ann_model(X_test_tensor)
    return torch.max(y_pred, 1)[1].numpy()

# Main function to compare baseline, logistic regression, and ANN
def compare_models(X, y, lambda_val=1.0, hidden_units=10, K=10, J=30):
    outer_cv = KFold(n_splits=K, shuffle=True, random_state=1)
    results = {}

    # Pairwise comparisons: Baseline vs Logistic, Baseline vs ANN, Logistic vs ANN
    for (model_name, model_a, model_b, get_preds_fn) in [
        ("Baseline vs LR", baseline(), logistic_regression(lambda_val), lambda X_train, y_train, X_test: model_b.fit(X_train, y_train).predict(X_test)),
        ("Baseline vs ANN", baseline(), ann(hidden_units), lambda X_train, y_train, X_test: get_ann_predictions(model_b, X_train, y_train, X_test)),
        ("LR vs ANN", logistic_regression(lambda_val), ann(hidden_units), lambda X_train, y_train, X_test: get_ann_predictions(model_b, X_train, y_train, X_test))
    ]:
        r_j = []
        for _ in range(J // K):
            for train_idx, test_idx in outer_cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Predictions from both models
                y_pred_a = model_a.fit(X_train, y_train).predict(X_test) if model_name != "Baseline vs ANN" else baseline().fit(X_train, y_train).predict(X_test)
                y_pred_b = get_preds_fn(X_train, y_train, X_test)

                # Calculate error rates
                error_a = 1 - accuracy_score(y_test, y_pred_a)
                error_b = 1 - accuracy_score(y_test, y_pred_b)
                r_j.append(error_a - error_b)

        # Perform the correlated t-test
        mean_diff, conf_interval, p_val = correlated_t_test(r_j, J)
        results[model_name] = {
            "mean_difference": mean_diff,
            "confidence_interval": conf_interval,
            "p_value": p_val
        }

    # Display results
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Mean Difference: {result['mean_difference']:.4f}")
        print(f"  Confidence Interval: {result['confidence_interval']}")
        print(f"  P-value: {result['p_value']:.4f}\n")

    # Plotting
    model_names = list(results.keys())
    mean_diffs = [results[name]["mean_difference"] for name in model_names]
    conf_intervals = [results[name]["confidence_interval"] for name in model_names]
    p_values = [results[name]["p_value"] for name in model_names]

    # Plot: Confidence Intervals for Mean Difference
    plt.figure(figsize=(8, 5))

    # Calculate the yerr array as (2, n) shape
    yerr = np.array([[mean_diffs[i] - conf_intervals[i][0], conf_intervals[i][1] - mean_diffs[i]] for i in range(len(model_names))]).T

    plt.errorbar(model_names, mean_diffs, yerr=yerr,
                fmt='o', color='black', ecolor='red', capsize=5)
    plt.ylabel("Mean Difference in Error Rate with 95% CI")
    plt.title("Confidence Intervals for Mean Differences in Error Rates")
    plt.xticks(rotation=15)
    plt.savefig("project2/figures/CI.png")

# Function to compute mean and confidence interval
def compute_mean_ci(errors, confidence=0.95):
    """
    Compute the mean and confidence interval of test errors across folds.
    
    Parameters:
        errors (list or np.array): List of test errors across folds.
        confidence (float): Confidence level for the interval.

    Returns:
        mean_error (float): Mean test error.
        ci (tuple): Confidence interval for the mean error.
    """
    errors = np.array(errors)
    mean_error = np.mean(errors)
    n_folds = len(errors)
    std_error = np.std(errors, ddof=1) / np.sqrt(n_folds)
    
    # Calculate confidence interval
    h = std_error * t.ppf((1 + confidence) / 2., n_folds - 1)
    ci = (mean_error - h, mean_error + h)
    
    return mean_error, ci

# Function to extract test errors for each model and calculate aggregated metrics
def aggregate_results(results_df):
    """
    Aggregate cross-validation errors for each model type and compute mean error and confidence intervals.
    
    Parameters:
        results_df (pd.DataFrame): DataFrame with 'method' and 'test_error' columns.
    
    Returns:
        aggregated_results (dict): Dictionary with mean and confidence intervals for each model.
    """
    aggregated_results = {}
    
    for model in results_df['method'].unique():
        model_errors = results_df[results_df['method'] == model]['test_error']
        mean_error, ci = compute_mean_ci(model_errors)
        aggregated_results[model] = {'mean_error': mean_error, 'ci': ci}
    
    return aggregated_results

# Function to plot mean test errors with confidence intervals
def plot_aggregated_results(aggregated_results):
    """
    Plot mean test error with confidence intervals for each model.
    
    Parameters:
        aggregated_results (dict): Dictionary with mean and confidence intervals for each model.
    """
    models = list(aggregated_results.keys())
    mean_errors = [aggregated_results[model]['mean_error'] for model in models]
    cis = [aggregated_results[model]['ci'] for model in models]

    # Calculate y-errors for confidence intervals
    yerr = [[mean_errors[i] - cis[i][0], cis[i][1] - mean_errors[i]] for i in range(len(models))]
    yerr = np.array(yerr).T  # Shape for error bars

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, mean_errors, yerr=yerr, capsize=5, color='skyblue', edgecolor='black')
    plt.ylabel("Mean Test Error with 95% CI")
    plt.title("Mean Test Error Across Folds with Confidence Intervals")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("project2/figures/mean_test_error.png")


# Example usage
# Assuming X, targets are your features and labels
compare_models(X, targets, lambda_val=0.001, hidden_units=200)

# Assuming results_df is the DataFrame output from run_two_level_cross_validation
aggregated_results = aggregate_results(results_df)
#plot_aggregated_results(aggregated_results)

