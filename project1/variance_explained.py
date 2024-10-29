from ucimlrepo import fetch_ucirepo 
from project1.PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def variance_explained(rho, cumulative_rho, threshold):
    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, "x-", color='midnightblue')
    plt.plot(range(1, len(cumulative_rho) + 1), cumulative_rho, "o-", color='slategray')
    plt.plot([1, len(rho)], [threshold, threshold], "k--")
    plt.title("Variance explained by principal components")
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained")
    plt.legend(["Individual", "Cumulative", "Threshold"])
    plt.grid()

    plt.savefig("figures/explained_variance.png")

