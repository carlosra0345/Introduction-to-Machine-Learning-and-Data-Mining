from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle  # Import shuffle function

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features
targets = np.concatenate(glass_identification.data.targets.to_numpy())
N, M = X.shape

X, targets = shuffle(X, targets, random_state=42)
