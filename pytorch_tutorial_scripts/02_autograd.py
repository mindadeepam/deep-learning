
'''
see https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html to understand better.
'''


import torch
import numpy as np


# requires_grad=True makes pytorch calculate a computation graph in which it includes any deriving tensor for easy backward computation.
# pytorch is dynamic. ie that grph is destroyed after every backward pass and reconstructed during forward pass.
# only float valued tensors can have gradents!
# x = torch.randn(3,2, requires_grad=True)

x=torch.tensor([5.0,4.0], requires_grad=True)
print(x)

y=x+1
print(y)            # since y is sum of x iwth something, it gets addition_backwards function

z=y*3
print(z)            # grad_fn=<MulBackward0>

w=z.mean()          # <MeanBackward0>
w
np.random.rand(3,2).mean(axis=0)    # axis 0 means innermost axis ie col wise in both torch and numpy

w.backward()

# w must be one element tensor/scalar. default arg of backward is 1 i.e loss.backward(1) bcuz generally our loss term
# or last term (on which we call .bcakward is a one element tensor, sum/mean/etc) if its not then we need to pass vector.shape=loss.shape to .backwards() 
# if our loss is [4,5] then we can pass loss.backward([1,2]) as weights for the loss!!

'''
loss ya w (here) is not passed directly backwards. when we define forward passes with tensors that
have tensors with req_grad=true, pytorch automatically maintains a computation graph which stores gradients(if req),
the grad_fucntion, leaf_node or not, etc.

it passes the value 1 here bcuz dE/dw will be gradient at w toh at E the gradient is dE/dE=1 
'''

# only leaf nodes store gradients, waise bhi parameters are always leaf nodes in any layer/model
print(w.grad)           # None
print(z.grad)           # None
print(y.grad)           # None
print(x.grad)           # some value

# to detach from comp graph/remove gradients from tensor-
x.requires_grad_(False)         # trailing_ mans inplace
x.detach()
with torch.no_grad():
    'something with x'
    print(x)



weights = torch.tensor([2,2,2], requires_grad=True, dtype=torch.float)

for epoch in range(3):
    output = (weights*3).sum()
    output.backward()
    print(weights.grad)         # gradients accumulate in each epoch if we dont use grad=0

    # optim.step() 
    weights.grad.zero_()


# example 
x = torch.tensor([3.0,2.0], requires_grad=True)
y=x*x*3                                             # (local) grad func, dy/dx = 6*x  
z=y+1                                               # loc. grad func, dz/dy=1 
E=z.sum()                                           # loc grad, dE/dz = 1
E.backward()                                        # dE/dE=1
print(x.grad)                                       
'''dE/dx = dE/dz * dz/dy * dy/dx = 1*1*6*x ----> x.grad = [18.0,12.0]

'''