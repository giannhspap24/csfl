import torch
import torch.nn.functional as F
from torch import nn,flatten
from torch.utils.data import DataLoader, Dataset
import os
import os.path
from PIL import Image

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df['path'][index]
        if not img_path or not os.path.exists(img_path):
            print(f"Invalid image path: {img_path}")
            raise FileNotFoundError(f"Image path {img_path} is not valid or does not exist.")

        try:
            #image = Image.open(img_path).resize((64, 64))
            X = Image.open(img_path).resize((64, 64))
            y = torch.tensor(int(self.df['target'][index]))
        except Exception as e:
            print(f"Failed to open image at path: {img_path}")
            raise e    
        
        if self.transform:
            X = self.transform(X)
        
        return X, y