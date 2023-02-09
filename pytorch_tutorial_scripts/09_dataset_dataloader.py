'''
# epoch = 1 forward and backward pass of all training samples
# batche_size = number of training samples in one foroward and backward pass
# number of iterations = number of passes, each using [batchsize] number of samples
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


# 1. prepare data
class WineDataset(Dataset):

    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0])    
        self.len = xy.shape[0]

    def __getitem__(self, idx):
        inputs = self.x[idx]        
        labels = self.y[idx]
        # inputs = torch.from_numpy(inputs)
        # labels = torch.from_numpy(inputs)
        return inputs, labels

    def __len__(self):
        return self.len  

    

filepath = './data/wine.csv'
dataset = WineDataset(filepath)
dataloader = DataLoader(dataset, shuffle=True, batch_size=32)


# 2. def model

class classifyWine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs



# input_dim = 13
# output_dim = 1
# model = classifyWine(input_dim, output_dim)


# 3. def loss, optim and other training params



# 4. training loop
for epoch in range(50):
    for batch in dataloader:
        inputs, labels = batch
        print(f"inputs[0] {inputs[0]}, \nlabels[0] {labels[0]}")
        break
    break