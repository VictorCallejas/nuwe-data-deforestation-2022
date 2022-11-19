import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

import torch

import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

from scipy.spatial import cKDTree
from scipy.special import softmax

IMAGE_SIZE = 240

DATASET_MEAN = [0.0593, 0.1184, 0.0867]
DATASET_STD = [0.0401, 0.0440, 0.0508]

K = 5

class NuweDataset(Dataset):
    '''Dataset class for the challenge
    Outputs the image, year and neighbours context(as seen on eda.ipynb)
    '''
    
    def __init__(self, data, directory, train=True):
        
        self.train = train
        self.data = data.reset_index(drop=True)
        self.directory = directory
        
        self.location_tree, self.neighbour_type = self.create_location_tree()
        
        self.transform = transforms.Compose([
                    transforms.Resize(IMAGE_SIZE),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.AutoAugment(AutoAugmentPolicy.IMAGENET),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DATASET_MEAN,
                                         std=DATASET_STD),
                    #transforms.RandomErasing()
                ])
        
        self.test_transform = transforms.Compose([
                    transforms.Resize(IMAGE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DATASET_MEAN,
                                         std=DATASET_STD)
                    ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        path  = self.directory + self.data.example_path[idx]
        image = Image.open(path)
            
        if self.train:
            image = self.transform(image)
            label = self.data.label[idx]
        else:
            image = self.test_transform(image)
            # For test predictions
            label = self.data.example_path[idx].split('/')[-1].split('.')[0]
            
        year = torch.tensor(self.data.year[idx] - 2001)
        neightbours_context = torch.tensor(self.get_neighbors_context(self.data.longitude[idx], self.data.latitude[idx]), dtype=torch.float32)
        
        return image, year, neightbours_context, label
    
    def get_neighbors_context(self, lon, lat):
        
        tmp = np.zeros(3)
        
        k_distance, k_idx = self.location_tree.query((lon, lat), k=K) # get neighbours and distances
        
        for n, distance in zip(k_idx, k_distance):
            #if distance < 500: ????? 
            tmp[self.neighbour_type[n]] = tmp[self.neighbour_type[n]] + 1
        
        tmp = softmax(np.array(tmp)) # Normalization
        
        return tmp
        
    
    @staticmethod
    def create_location_tree():
        
        train = pd.read_csv('data/raw/train.csv')
        
        lat = np.expand_dims(train.latitude.values, 1)
        long = np.expand_dims(train.longitude.values, 1)

        X = np.concatenate((lat, long), axis = 1)

        spatialTree = cKDTree(X)
        
        return spatialTree, train.label
