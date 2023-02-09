# 1. design model (input, output, forward pass)
# 2. construct loss and optimizer
# 3. Training loop
#       - forward pass: compute prediction, loss
#       - backward pass: compute gradients
#       - update weights, reset gradients 


# f(x) = y = x**2
import torch
import torch.nn as nn

x = torch.tensor([1.0, 3.0, 2.5, 5.0])
y = torch.tensor([1.0, 9.0, 6.25, 25.0])
w = torch.tensor(1.0, requires_grad=True)

def forward(x):
    return x*w

# def loss(y, y_predicted):
#     return ((y_predicted-y)**2).mean()

def gradients():
    pass

lr=0.01
n_iter=100
criterion = nn.MSELoss()
optim = torch.optim.SGD(params=[w], lr=lr)

print(f"Before training: f(5) = {forward(torch.tensor(5))}")

for epoch in range(n_iter):
    # forward pass
    y_predicted = forward(x)
    loss = criterion(y, y_predicted)
    
    # backward pass
    loss.backward()

    # update weights
    optim.step()

    # reset gradients
    optim.zero_grad()

    print(f"epoch: {epoch+1}, loss: {loss:.3f}, w: {w:.3f}")

print(f"After training: f(5) = {forward(torch.tensor(5))}")