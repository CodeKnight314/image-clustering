import torch
import torchvision.models as models
import numpy as np
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, n_cluster, device="cuda"):
        self.device = torch.device(device) 
        self.model = models.resnet18(weights="DEFAULT").to(self.device)
        self.model.eval() 
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.kmeans = KMeans(n_clusters=n_cluster)
        
        self.n_cluster = n_cluster
        
    def extract_features(self, dataloader): 
        features = []
        with torch.no_grad(): 
            for images, _ in dataloader: 
                images = images.to(self.device)
                output = self.model(images)
                output = output.view(output.size(0), -1)
                features.append(output.cpu().numpy())
        return np.vstack(features)
    
    def fit(self, dataloader): 
        features = self.extract_features(dataloader)
        self.kmeans.fit(features)
        labels = self.kmeans.labels_
        return labels, features