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
    def transform(self, num_components=None):
        if num_components is None:
            num_components = self.Vt.shape[0]  # Default to all components
        
        # Select the top `num_components` principal components
        selected_components = self.Vt[:num_components]
        
        # Project the data onto the selected components
        projected_data = np.dot(self.features, selected_components.T)
        return projected_data