import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from models import ann, logistic_regression, baseline, train_neural_net
from dataset import glass_identification, X, targets
from two_level_CV import two_level_cv_ann, two_level_cv_baseline, two_level_cv_logistic_regression
from itertools import combinations
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Ensure targets are zero-indexed
if np.min(targets) == 1:
    targets = targets - 1  # Convert 1-based to 0-based indexing if necessary
n_classes = len(np.unique(targets))

# Function to find the two most important features based on inter-class distances
def find_most_distinctive_features(X, y):
    n_features = X.shape[1]
    class_labels = np.unique(y)
    
    best_pair = None
    best_score = -np.inf

    # Iterate over each pair of features
    for feature1, feature2 in combinations(range(n_features), 2):
        feature_pair = X[:, [feature1, feature2]]
        
        # Calculate class centroids
        class_centroids = np.array([
            feature_pair[y == label].mean(axis=0) for label in class_labels
        ])
        
        # Compute mean pairwise distance between class centroids
        inter_class_distances = np.linalg.norm(
            class_centroids[:, np.newaxis] - class_centroids, axis=2
        )
        mean_distance = np.mean(inter_class_distances)
        
        # Update best pair if this one has a higher score
        if mean_distance > best_score:
            best_score = mean_distance
            best_pair = (feature1, feature2)
    
    return best_pair

# Function to plot each model's predictions based on the two most important features with colored zones
def plot_model_with_zones(X, y, model, feature1_idx, feature2_idx, title):
    # Extract the two most important features for plotting purposes
    X_plot = X[:, [feature1_idx, feature2_idx]]
    y = np.array(y)

    # Create a mesh grid for plotting decision boundaries
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict on the grid using the full feature set but focusing on selected features for visualization
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            # Tile the selected features across the entire grid
            full_grid = np.repeat(np.mean(X, axis=0)[np.newaxis, :], grid_points.shape[0], axis=0)
            full_grid[:, [feature1_idx, feature2_idx]] = grid_points
            grid_tensor = torch.tensor(full_grid, dtype=torch.float32)
            preds = model(grid_tensor).argmax(dim=1).numpy()
    else:
        # Logistic Regression or Baseline model
        full_grid = np.repeat(np.mean(X, axis=0)[np.newaxis, :], grid_points.shape[0], axis=0)
        full_grid[:, [feature1_idx, feature2_idx]] = grid_points
        preds = model.predict(full_grid)

    preds = preds.reshape(xx.shape)

    # Define a color map for the decision boundary zones with lighter colors
    cmap_light = ListedColormap([plt.cm.viridis(i / n_classes, alpha=0.3) for i in range(n_classes)])
    cmap_bold = ListedColormap([plt.cm.viridis(i / n_classes) for i in range(n_classes)])

    # Plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, preds, cmap=cmap_light, alpha=0.6)  # Lighter color zones for each class

    # Scatter plot of the data points with true class colors
    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=50)
    plt.colorbar(scatter, ticks=range(n_classes), label="Class")
    plt.xlabel(f'Feature {feature1_idx + 1}')
    plt.ylabel(f'Feature {feature2_idx + 1}')
    plt.title(title)
    plt.savefig(f"project2/figures/plot_{str(title)}.png")


# Find the most distinctive feature pair
feature1_idx, feature2_idx = find_most_distinctive_features(X, targets)
#feature1_idx = 7
#feature2_idx = 0

# Define hyperparameters
best_lambda = 0.001
best_h = 50

# Train models using all features
logistic_model = logistic_regression(best_lambda)
trained_ann_model, _ = train_neural_net(ann(best_h), F.cross_entropy, X, targets)
baseline_model = baseline()

# Train each model
logistic_model.fit(X, targets)
trained_ann_model, _ = train_neural_net(ann(best_h), F.cross_entropy, X, targets)
baseline_model.fit(X, targets)

# Plot with colored zones for each model
plot_model_with_zones(X, targets, logistic_model, feature1_idx, feature2_idx, "Logistic Regression")
plot_model_with_zones(X, targets, trained_ann_model, feature1_idx, feature2_idx, "Artificial Neural Network")
plot_model_with_zones(X, targets, baseline_model, feature1_idx, feature2_idx, "Baseline")
