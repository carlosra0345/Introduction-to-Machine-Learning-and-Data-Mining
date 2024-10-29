from sklearn.preprocessing import StandardScaler
import numpy as np

class PCA:
    def __init__(self, features):
        self.features = self.normalize_matrix(features) 
        self.U, self.S, self.Vt = self.SVD()
        
    def normalize_matrix(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data
        
    def SVD(self):
        return np.linalg.svd(self.features)
    
    # Get principal components
    def principal_components(self):
        return self.Vt.T

    # Project data onto the principal components
    def transform(self, first, second):
        # Convert inputs to integers to use them as indices
        first = int(first) - 1  
        second = int(second) - 1
        
        # Ensure the provided components are within valid bounds
        if first < 0 or first >= self.Vt.shape[0] or second < 0 or second >= self.Vt.shape[0]:
            raise ValueError("Invalid principal component indices")
        
        # Select the corresponding principal component vectors (columns) from Vt
        selected_components = self.Vt[[first, second], :]
        
        # Project the data onto the selected principal components
        projected_data = np.dot(self.features, selected_components.T)
        
        return projected_data