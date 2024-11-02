from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features
targets = np.concatenate(glass_identification.data.targets.to_numpy())
N, M = X.shape

#SHUFFLE EVERYTHING