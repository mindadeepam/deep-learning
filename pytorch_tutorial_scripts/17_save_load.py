'''
1. these will save and load models using pickle. lazy way. not recommended.

torch.save(model, PATH)                              # this can save any dictionary to given path
model = torch.load(PATH)
model.eval()    


2. this will save only weights and biases

torch.save(model.state_dict(), PATH)

model = Model(*args, **kwargs)                       # model must be created again with params
model.load_state_dict(torch.load(PATH))
model.eval()
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

model = Model(input_dim = 10)

inputs = torch.randn(5,10, dtype=torch.float32)
model(inputs)

FILE = './pytorch_tutorial_scripts/model.pth'
# torch.save(model, FILE)
# loaded_model = torch.load(FILE)

torch.save(model.state_dict(), FILE)
loaded_model = Model(10)                #load model from params and class
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()


# optimizer also has a state dict that stores its params
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.state_dict()


## To Store CHECKPOINTS during training -  (checkpoint is a dict with these kv pairs )

checkpoint = {
    'epoch': 5,
    'model_state' : model.state_dict(),
    'optim_state' : optimizer.state_dict()
}

# torch.save(checkpoint, './pytorch_tutorial_scripts/checkpoint.pth')
loaded_checkpoint = torch.load('./pytorch_tutorial_scripts/checkpoint.pth')
#load model with same params
model = Model(10)
# load optimizer with same params
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']


## GPU considerations (pass map_loaction argument when save and load is on different devices)

# 1 save on gpu during training, load on cpu

    # save in gpu
device = torch.device('cuda')
model.to(device)
torch.save(model.state_dict(), FILE)

    # load in cpu
device = torch.device('cpu')
model = Model(10)
model.load_state_dict(torch.load(FILE, map_location=device))
model.to(device)

# 2 save on cpu load on gpu

device = torch.device('cpu')
model.to(device)
torch.save(model.state_dict(), FILE)

device = torch.device('cuda')
model = Model(10)
model.load_state_dict(torch.load(FILE, map_location="cuda:0"))
model.to(device)

# 3 save on gpu load on gpu

torch.save(model.state_dict(), FILE)

model=Model(10)
model.load_state_dict(torch.load(FILE))
model.to(device)



