
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset

class CANDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = Path(root_dir) / ('train' if is_train else 'val')
        self.is_train = is_train
        self.transform = transform
        self.total_size = len(os.listdir(self.root_dir))
            
    def __getitem__(self, idx):
        filename = f'{idx}.npz'
        filename = self.root_dir / filename
        data = np.load(filename)
        X, y = data['X'], data['y']
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_tensor = torch.unsqueeze(X_tensor, dim=0)
        y_tensor = torch.tensor(y, dtype=torch.long)
        if self.transform:
            X_tensor = self.transform(X_tensor)
        return X_tensor, y_tensor
    
    def __len__(self):
        return self.total_size