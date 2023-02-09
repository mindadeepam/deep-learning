
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from time import time
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## 1. prepare data

# here arbitrary values have been used, calc mean and std of your own data or use imagenet's values
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

img_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}

batch_size = 8
data_dir = './pytorch_tutorial_scripts/data/hymenoptera_data/'
img_datasets = {x: ImageFolder(os.path.join(data_dir, x), transform=img_transforms[x])
                for x in ['train', 'val']
}
img_loaders = {x: DataLoader(img_datasets[x], batch_size=batch_size, shuffle=True, )
                for x in ["train", "val"]
}
datasize = {x: len(img_datasets[x]) for x in ['train', 'val']}
class_names = img_datasets['train'].classes

# get a batch of inputs and labels
samples, labels = next(iter(img_loaders['train']))

def show_image(img, title=None):
    # plt.imshow requires H,W,num_channels. to change from CHW to HWC use np transpose or torch permute
    img = torch.permute(img, (1,2,0))
    # de normalize img
    img = img*std + mean
    plt.imshow(img)
    plt.title(title)
    plt.show()


# see an image
show_image(samples[0])

# see a grid of images
grid = torchvision.utils.make_grid(samples)
show_image(grid, [class_names[x] for x in labels])



## 2. define training loop

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    phases = ["train", "test"]
    best_loss = float('inf')

    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_correct = 0
            if phase=='train': 
                scheduler.step()
            # Iterate over data.
            for i, batch in enumerate(img_loaders[phase]):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                # only calc gradients in training
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, y_pred = torch.max(outputs, axis=1)
                
                # backward + optimize only if in training phase
                    if(phase=='train'):
                        optimizer.zero_grad()
                        loss.backward()
                        print("here")
                        optimizer.step()
                        print("here")

                running_loss += loss.item()*inputs.size[0]
                running_correct += y_pred.eq(labels).sum().item()

            # step scheduler after epochs


            epoch_loss = running_loss/datasize[phase]
            epoch_acc = running_correct/datasize[phase]

            print(f'epoch {epoch}, phase {phase}, loss {epoch_loss:.3f}, acc {epoch_acc:.3f}')

            # update best_loss and save best model weights
            if phase=='val' and epoch_loss<best_loss:
                best_loss=epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())

    # load best model
    model.load_state_dict(best_model_weights)
    return model


## 3.a Finetune the model by training all model weights after

# load model
model = torchvision.models.resnet18(pretrained=True)

# change classification head
last_layer_input_features = model.fc.in_features
model.fc = nn.Linear(last_layer_input_features, 2)
model.to(device)

# define loss, optim, sched ..
lr = 0.01
criterion = nn.CrossEntropyLoss()
# all parameters are passed to optimizer to update weights
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
num_epochs = 10

train_model(model, criterion, optimizer, scheduler, num_epochs)
