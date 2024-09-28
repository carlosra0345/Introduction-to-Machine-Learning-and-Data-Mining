from ucimlrepo import fetch_ucirepo 
from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np

figure_file = 'figures/' + input('Enter name of file to save figure: ')
num_desired_components = int(input('Enter desired number of components to reduce to: '))

  
# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features 
targets = np.concatenate(glass_identification.data.targets.to_numpy())

pca = PCA(X)

projected_data = pca.transform(num_components=num_desired_components)

# Create a color map based on unique targets
unique_targets = np.unique(targets)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_targets)))  # Create a color map
target_color_map = {target: color for target, color in zip(unique_targets, colors)}

# Map targets to colors
target_colors = np.array([target_color_map[target] for target in targets])

# Create a scatter plot with the mapped colors
plt.figure(figsize=(10, 8))
plt.scatter(projected_data.T[0], projected_data.T[1], c=target_colors, edgecolor='k', s=100)

# Add a title and labels
plt.title('PCA of Glass Identification Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)

# Create a custom legend for unique targets
handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(target), 
                       markerfacecolor=target_color_map[target], markersize=10) 
           for target in unique_targets]
plt.legend(handles=handles, title='Targets', loc='best')

# Save the plot
plt.savefig(figure_file)

summ = 0
for n in pca.S:
   summ += n*n 
var = []
for i in range(1,10):
    temp = 0
    for num in pca.S[:i]:
        temp += num*num
    var.append(temp/summ)
    
plt.plot(var)
plt.savefig('figures/fraction-thingy.png')