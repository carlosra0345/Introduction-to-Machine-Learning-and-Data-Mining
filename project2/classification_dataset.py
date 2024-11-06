from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle  # Import shuffle function
from sklearn.preprocessing import StandardScaler

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features
targets = np.concatenate(glass_identification.data.targets.to_numpy())
N, M = X.shape

# standardizing dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

X, targets = shuffle(X, targets, random_state=1000)

