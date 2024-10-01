from ucimlrepo import fetch_ucirepo 
from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from variance_explained import*
from variance_coef import*
from principle_components import*
from pearplot import*
from multiplot import*
import seaborn as sns
import pandas as pd

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features
targets = np.concatenate(glass_identification.data.targets.to_numpy())
N, M = X.shape

# get the PCA of the normalized matrix
pca = PCA(X)

# compute SVD of X
U, S, V = pca.U, pca.S, pca.Vt

# compute variance explained by principal components
rho = (S * S) / (S * S).sum()
cumulative_rho = np.cumsum(rho)
threshold = 0.9

# compute how many principle componants needed for 90% and 95% variance
components_90 = np.argmax(cumulative_rho >= 0.90) + 1
components_95 = np.argmax(cumulative_rho >= 0.95) + 1
attributeNames = X.columns.tolist()

first_PC = input('Enter the first principle component: ')
second_PC = input('Enter the second principle component: ')

principal_components(first_PC, second_PC, pca, targets)
variance_explained(rho, cumulative_rho, threshold)
variance_coef(V, components_90, attributeNames)
pearplot(pca, targets, 5)
multiplot(X, attributeNames, pca, N, targets)