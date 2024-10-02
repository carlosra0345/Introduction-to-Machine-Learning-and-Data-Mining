from ucimlrepo import fetch_ucirepo 
from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import seaborn as sns
import pandas as pd

def variance_coef(V, components_90, attribute_names):
    # Plotting of variance coefficient
    pcs = np.arange(components_90 + 1)
    legendStrs = ["PC" + str(e + 1) for e in pcs]
    
    # Define a color palette
    colors = sns.color_palette("cividis", n_colors=len(pcs))
    
    bw = 0.1  # Width of the bars
    r = np.arange(1, len(attribute_names) + 1)  # Adjust based on the number of attributes
    plt.clf()
    
    for i in pcs:
        if i < V.shape[1]:
            plt.bar(r + i * bw, V[:, i][:len(attribute_names)], width=bw, color=colors[i], alpha=0.8, label=legendStrs[i])
    
    # Set x-ticks to actual attribute names
    plt.xticks(r + bw * (len(pcs) - 1) / 2, attribute_names)  
    plt.xlabel("Attributes", fontsize=12)
    plt.ylabel("Component Coefficients", fontsize=12)
    plt.title("Glass Indentification: PCA Component Coefficients", fontsize=14)
    
    # Improve legend placement
    plt.legend(title="Principal Components", loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Adjust the coordinates as needed
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig("figures/var_coef.png")  # Save the plot