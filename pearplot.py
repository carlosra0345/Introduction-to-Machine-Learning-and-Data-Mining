from ucimlrepo import fetch_ucirepo 
from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.linalg import svd

def pearplot(pca, targets, num_components):
    # Create a DataFrame for principal components
    pc_data = pd.DataFrame()

    # Project the data onto the first 'num_components' principal components
    for i in range(1, num_components + 1):
        pc_data[f'PC{i}'] = pca.transform(i, 1)[:, 0]  # Capture each principal component

    # Add the target labels to the DataFrame
    pc_data['Target'] = targets

    # Create the pairplot using seaborn with custom plot settings for scatter points
    sns.pairplot(pc_data, hue='Target', palette='cividis', diag_kind='kde',
                 plot_kws={'alpha': 0.75, 'edgecolor': 'none', 's': 50})  # Adjust alpha and remove edge color
    
    # Save the updated plot
    plt.savefig("figures/pearplot.png")

