import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import os
from glob import glob
from torchvision import transforms
from PIL import Image


class chestXrayDataset(Dataset):
    def __init__(self, img_path, datainfo_path):
        self.img_path = img_path
        self.datainfo_path = datainfo_path
        self.data_csv = pd.read_csv(self.datainfo_path)

        self.label_map  = {"PA": 0, "AP": 1}
        
        self.images = glob(os.path.join(self.img_path))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB normalization
        ])


        img_filenames = [os.path.basename(img_path) for img_path in self.images]
        self.matching_rows = self.data_csv[self.data_csv['Image Index'].isin(img_filenames)]
        self.img_labels = self.matching_rows['View Position']


    def __len__(self):
        return len(self.images[:100])
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_str = self.img_labels[idx]
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        label = self.label_map.get(label_str)

        return img_tensor, label