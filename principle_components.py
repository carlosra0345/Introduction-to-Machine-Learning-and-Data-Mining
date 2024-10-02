from ucimlrepo import fetch_ucirepo 
from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def principal_components(first_PC, second_PC, pca, targets):
    projected_data = pca.transform(first_PC, second_PC)

    # Create a color map based on unique targets
    unique_targets = np.unique(targets)
    colors = plt.cm.cividis(np.linspace(0, 1, len(unique_targets)))  # Create a color map
    target_color_map = {target: color for target, color in zip(unique_targets, colors)}

    # Map targets to colors
    target_colors = np.array([target_color_map[target] for target in targets])

    # Create a scatter plot with the mapped colors
    plt.figure(figsize=(10, 8))
    plt.scatter(projected_data.T[0], projected_data.T[1], c=target_colors, edgecolor='k', s=100)

    # Add a title and labels
    plt.title('PCA of Glass Identification Dataset')
    plt.xlabel(f"Principal Component {first_PC}")
    plt.ylabel(f"Principal Component {second_PC} ")
    plt.grid(True)

    # Create a custom legend for unique targets
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(target), 
                        markerfacecolor=target_color_map[target], markersize=10) 
            for target in unique_targets]
    plt.legend(handles=handles, title='Targets', loc='best')

    # Save the plot
    plt.savefig("figures/pca.png")
