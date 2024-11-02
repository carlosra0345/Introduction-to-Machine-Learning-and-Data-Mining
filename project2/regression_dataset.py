from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle  # Import shuffle function
from sklearn.preprocessing import StandardScaler

# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 

data = glass_identification.data.features

# seperating refreactive index as target variable and remaining oxidation attributes as predictors
y = data['RI'].values
X = data.drop(columns=['RI']).values

scaler = StandardScaler()
x_standardized = scaler.fit_transform(X)

X, targets = shuffle(x_standardized, y, random_state=42)
