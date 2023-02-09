import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import numpy as np

class WineDataset(Dataset):
    def __init__(self, path, transforms=None):
        xy = np.loadtxt(path, delimiter=',', skiprows=1)
        self.x = xy[:,1:]
        self.y = xy[:,0]
        self.len = xy.shape[0]
        self.transforms = transforms

    def __getitem__(self, idx):
        samples = self.x[idx], self.y[idx]
        if self.transforms:
            samples = self.transforms(samples)
        
        return samples

    def __len__(self):
        return self.len


class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)
class someotherTransform:
    def __call__(self, sample):
        return sample

composed = Compose([ToTensor(), someotherTransform])

dataset = WineDataset("./tutorial/wine.csv", transforms=composed)
dataloader = DataLoader(dataset, shuffle=True, batch_size=8)            # totensor is by default it seems

for epoch in range(1):
    for batch in dataloader:
        inputs, labels = batch
        print(inputs, "\n", labels)
        break

