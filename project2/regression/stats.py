import numpy as np
from scipy import stats

ann = np.array([0.1477, 0.0472, 0.1647, 0.0855, 0.2545, 0.2475, 0.0873, 0.0454, 0.1291, 0.2308])
lr = np.array([0.1196, 0.0331, 0.0946, 0.0704, 0.2207, 0.2874, 0.0236, 0.0305, 0.1184, 0.1679])
baseline = np.array([1.2849, 0.6012, 1.2857, 0.3783, 1.1313, 1.8683, 0.6985, 1.2128, 0.6606, 1.0063])

def correlated_t_test(x, y):
    """
    Perform the correlated t-test for two models' cross-validation results.
    """
    # Calculate the differences
    d = x - y
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)
    n = len(d)
    
    # Calculate correlation between the two sets of errors
    r = np.corrcoef(x, y)[0, 1]
    
    # Compute the t-statistic for the correlated t-test
    t_stat = d_mean / (d_std * np.sqrt((1 - r) / n))
    
    # Calculate p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    # Compute the confidence interval (95%)
    confidence_interval = stats.t.interval(0.95, df=n-1, loc=d_mean, scale=d_std * np.sqrt((1 - r) / n))
    
    return d_mean, t_stat, p_value, confidence_interval

results_ann_vs_lr = correlated_t_test(ann, lr)
results_ann_vs_baseline = correlated_t_test(ann, baseline)
results_lr_vs_baseline = correlated_t_test(lr, baseline)

print("ANN vs LR:")
print(f"Mean difference: {results_ann_vs_lr[0]:.4f}")
print(f"t-statistic: {results_ann_vs_lr[1]:.4f}")
print(f"p-value: {results_ann_vs_lr[2]:.4f}")
print(f"95% CI: [{results_ann_vs_lr[3][0]:.4f}, {results_ann_vs_lr[3][1]:.4f}]")

print("\nANN vs Baseline:")
print(f"Mean difference: {results_ann_vs_baseline[0]:.4f}")
print(f"t-statistic: {results_ann_vs_baseline[1]:.4f}")
print(f"p-value: {results_ann_vs_baseline[2]:.4f}")
print(f"95% CI: [{results_ann_vs_baseline[3][0]:.4f}, {results_ann_vs_baseline[3][1]:.4f}]")

print("\nLR vs Baseline:")
print(f"Mean difference: {results_lr_vs_baseline[0]:.4f}")
print(f"t-statistic: {results_lr_vs_baseline[1]:.4f}")
print(f"p-value: {results_lr_vs_baseline[2]:.4f}")
print(f"95% CI: [{results_lr_vs_baseline[3][0]:.4f}, {results_lr_vs_baseline[3][1]:.4f}]")
