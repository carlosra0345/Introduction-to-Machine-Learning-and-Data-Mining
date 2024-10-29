from ucimlrepo import fetch_ucirepo 
from project1.PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.linalg import svd

def multiplot(X, attributeNames, pca, N, targets):
    classNames = sorted(set(targets))  # Unique class names from the targets
    classDict = dict(zip(classNames, range(len(classNames))))  # Create a mapping from class names to numeric labels

    # Extract vector y, convert to NumPy array using the class dictionary
    y = np.asarray([classDict[value] for value in targets])

    # Subtract the mean from the data
    Y1 = X - X.mean(axis=0)  # Zero-mean dataset

    # Standardized dataset
    Y2 = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardize

    # Store the two datasets in a list
    Ys = [Y1, Y2]
    titles = ["Zero-mean", "Zero-mean and unit variance"]
    threshold = 0.9
    i = 0
    j = 1

    # Make the plot
    plt.figure(figsize=(15, 10))  # Adjusted size for better spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust spacing between subplots
    plt.title("Glass Identification: Effect of Standardization")

    for k in range(2):
        # Obtain the PCA solution by calculating the SVD of either Y1 or Y2
        U, S, Vh = svd(Ys[k], full_matrices=False)
        V = Vh.T  # Transpose to get V

        # Compute the projection onto the principal components using Z = S * V
        Z = U * S  # This should have the shape (N, k) where k is the number of components

        # Plot projection
        plt.subplot(3, 2, 1 + k)
        C = len(classNames)
        colors = plt.cm.cividis(np.linspace(0, 1, C))  # Use plasma color palette for classes
        for c in range(C):
            plt.plot(Z[y == c, i], Z[y == c, j], ".", color=colors[c], alpha=0.6, label=classNames[c])
        plt.xlabel("PC" + str(i + 1))
        plt.ylabel("PC" + str(j + 1))
        plt.title(titles[k] + "\nProjection")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')
        plt.axis("equal")

        # Plot attribute coefficients in principal component space
        plt.subplot(3, 2, 3 + k)
        for att in range(V.shape[1]):
            plt.arrow(0, 0, V[att, i], V[att, j], color='black', alpha=0.7)
            plt.text(V[att, i], V[att, j], attributeNames[att], fontsize=9)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel("PC" + str(i + 1))
        plt.ylabel("PC" + str(j + 1))
        plt.grid()

        # Add a unit circle
        plt.plot(np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01)), color='gray', linestyle='--', alpha=0.5)
        plt.title(titles[k] + "\nAttribute Coefficients")
        plt.axis("equal")

        # Plot cumulative variance explained
        plt.subplot(3, 2, 5 + k)
        rho = (S * S) / (S * S).sum()  # Recompute variance explained
        plt.plot(range(1, len(rho) + 1), rho, "x-", color='midnightblue')
        plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", color='slategray')
        plt.plot([1, len(rho)], [threshold, threshold], "k--")
        plt.title("Variance Explained by Principal Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.legend(["Individual", "Cumulative", "Threshold"], loc='upper right', fontsize='small')
        plt.grid()

    plt.savefig("figures/multiplot.png")  # Save the plot
