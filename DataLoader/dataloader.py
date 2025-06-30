import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
from glob import glob

class chestXrayDataset(Dataset):
    def __init__(self, img_path, datainfo_path, transform=None, scaler=None):
        self.img_path = img_path
        self.datainfo_path = datainfo_path
        self.data_csv = pd.read_csv(self.datainfo_path)
        
        self.transform = transform
        self.scaler = scaler

        self.images = glob(os.path.join(self.img_path))

        img_filenames = [os.path.basename(img_path) for img_path in self.images]
        self.matching_rows = self.data_csv[self.data_csv['Image Index'].isin(img_filenames)]
        self.img_labels = self.matching_rows['View Position']


    def __len__(self):
        return len(self.images)
    
