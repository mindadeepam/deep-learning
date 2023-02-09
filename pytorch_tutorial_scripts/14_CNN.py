
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
from torch.utils.data import DataLoader
import os

device='cuda' if torch.cuda.is_available() else 'cpu'

# prepare data
batch_size = 32
basepath = './tutorial/'

train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(basepath, 'data'), download=True,
                train=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(basepath, 'data'), download=True, 
                train=False, transform=transforms.ToTensor())

trainloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, )
testloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, )

itertrain = iter(trainloader)
samples = next(itertrain)
inputs, labels = samples
inputs.shape, labels.shape          
''' 
1 image = 3*32*32, 1 batch = 32*3*32*32
batch_size * num_channels * img_h * img_w
'''

# define model
class ConvNet(nn.Module):
    def __init__(self, input_channels=3, ):
        super().__init__()
        
        # feature extraction block
        '''conv layers me next ka input_channels must be equal to previous' num of output_channels; 
        they adjust image size params(width and height) themselves which is
        output width and height =  “𝑊′=(W-F+2P/S)+1”  where
        W - image width/input width,
        K - kernel size,
        P - padding,
        S - stride.”'''
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)           
        ''' 
        output size = batchsize*outchannels*W, 
        out W = 32-5+(2*0/1) + 1 = 28
        out_batch shape = 32*6*28*28
        '''
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # classificaton block
        '''to go from conv to fc, 
        num of neurons in fc = chan*W*H of output of prev'''
        self.fc1 = nn.Linear(16*5*5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten (batch_size , chan*W*H)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet()
model.to(device)

# define loss and optimizer
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
n_iter = 5

# define training loop
best_loss = float('inf')
for epoch in range(n_iter):
    print(f"Epoch {epoch+1}")
    # training loop
    total_loss = 0
    for i, batch in enumerate(trainloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)           # we get mean loss of batch

        # backward pass, wieght update and rest gradients
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss+=loss
        if i%500==0:
            print(f"\tepoch {epoch+1}/{n_iter}, step {i+1}/{len(trainloader)},  loss {loss:.3f}") 
        
    train_loss = total_loss/len(trainloader)
    print(f'train_loss {train_loss:.3f}')


    # eval loop
    total_test_loss = 0
    n_correct = 0 
    n_samples = 0 
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)           # we get mean loss of batch
            total_test_loss+=loss

            
            _, y_pred = torch.max(outputs, axis=1)      # values, indices
            correct = y_pred.eq(labels).sum().item()
            n_correct += correct
            n_samples += labels.shape[0]

        test_acc = n_correct/n_samples
        test_loss = total_test_loss/len(testloader)

        print(f'test_loss {test_loss:.3f}, test acc {test_acc:.3f}')

        if(test_loss<best_loss):
            best_loss=test_loss
            print(f'New best loss = {best_loss:.3f}')
        else:
            print("test performance is decreasing. Stopping training noww!!!")
            break



# test class-wise
n_correct_classwise = [0 for i in range(10)]
n_samples_classwise = [0 for i in range(10)]

with torch.no_grad():
    for i, batch in enumerate(testloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        _, y_pred = torch.max(outputs, axis=1)
        correct = y_pred.eq(labels).sum().item()
        for i in range(labels.shape[0]):
            label = labels[i]
            n_samples_classwise[label] += 1
            if y_pred[i]==label:
                n_correct_classwise[label]+=1
        

    print("Accuracy table")
    for i in range(10):
        class_acc = 100*n_correct_classwise[i]/n_samples_classwise[i]
        print(f"\tclass {i+1} - {class_acc} %")

