
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib
from torch.utils.data import DataLoader
import os
from torchvision import transforms

device='cuda' if torch.cuda.is_available() else 'cpu'


batch_size = 32
input_channels = 3
input_size = 28*28
hidden_size = 100
num_classes = 10

# dummy
images = torch.randn(batch_size, input_channels, 32,32) 


# model
conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)           
'''num of input_channels of next conv layer must be equal to previous conv layer's num of output_channels; 
they adjust image size params(width and height) themselves which is
output width and height =  “𝑊′=(W-F+2P/S)+1”  where
W - input width,
K - kernel size,
P - padding,
S - stride.”'''

pool = nn.MaxPool2d(2,2)        
# arguments are kernel size=2, and stride=2 (same as kernel size by default), dilation=1 by default
# a tuple can be passed for all arguments when H!=W
'''
arguments are kernel size=(2,2)here, and stride (same as kernel size by default), dilation=1 by default
output channels stay the same.
for shape = batch * chn * H * W
    Hout = GIF(1 + (Hin + 2*Pad[0] - (dilation[0] *(kernel_size[0]-1)) - 1)/stride[0])
    Wout = GIF(1 + (Win + 2*Pad[1] - (dilation[1] *(kernel_size[1]-1)) - 1)/stride[1])

when kernel_size=(n,n) no other param is passed 
    Wout = 1 + (Win - kernel_size)/kernel_size
'''
conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

# classificaton block
'''
to go from conv to fc, 
num of neurons in fc = chan*W*H of output of prev
'''
fc1 = nn.Linear(16*5*5, 128)
fc2 = nn.Linear(128, 64)
fc3 = nn.Linear(64,10)



# pass thorugh model
print(images.shape)

x = conv1(images)
print(x.shape)
''' 
output size = batchsize*outchannels*W, 
out W = 32-5+(2*0/1) + 1 = 28
out_batch shape = 32*6*28*28
'''

x = pool(x)
print(x.shape)
'''
Wout = 1 + (28-2)/2 = 14
'''

x = conv2(x)
print(x.shape)
'''
Wout = 14-5+0+1 = 10
out_batch shape = 32*16*10*10 
'''
x = pool(x)
print(x.shape)
'''outshape = 32*16*5*5'''

# flatten
x = x.view(-1, 16*5*5)

# linear
x = fc1(x.view(-1, 16*5*5))
x = fc2(x)
x = fc3(x)
print(x.shape)