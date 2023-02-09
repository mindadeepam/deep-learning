import numpy as np
import torch

# Approximate a function: y = f(x) = 2*x
x=torch.tensor([1,2,3,4], dtype=torch.float16)
y=torch.tensor([2,4,6,8], dtype=torch.float16)

w = torch.tensor([0.0], requires_grad=True)

# model prediction
def forward(x):
    return w*(x)

# loss = MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

print(f"Prediction before trainng: f(5) = {forward(torch.tensor([5.0])).item():.3f}")


lr = 0.01
n_iters=20

for epoch in range(n_iters):
    # forward pass
    y_predicted = forward(x)
    
    # loss
    l = loss(y, y_predicted)
    
    # gradients = backward pass
    l.backward()

    # update weights 
    with torch.no_grad(): 
        w -= lr*w.grad
    
    # zero gradients
    w.grad.zero_()
    
    if epoch%1==0:
        print(f"epoch {epoch+1}, w = {w}, loss = {l:.3f}")


print(f"\nPrediction after trainng: f(5) = {forward(torch.tensor(5.0)).item():.3f}")
