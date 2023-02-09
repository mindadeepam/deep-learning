# 1. design model (input, output, forward pass)
# 2. construct loss and optimizer
# 3. Training loop
#       - forward pass: compute prediction, loss
#       - backward pass: compute gradients
#       - update weights, reset gradients 


# y = x*4 + 2

import torch
import torch.nn as nn 

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float)
y = torch.tensor([[6],[10],[14],[18]], dtype=torch.float)

n_samples, n_features = x.shape
input_shape = n_features
output_shape = 1

# custom model def
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # def layers
        self.lin = nn.Linear(input_dim, output_dim)    
    
    def forward(self, x):
        return self.lin(x)

# model = nn.Linear(input_shape, output_shape)
model = LinearRegression(input_shape, output_shape)

# training-params
lr=0.01
n_iter=20
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)          # model.parameters() gives us all leaf nodes/ie tunable parameters

print(f"Before training, f(5) = {model(torch.tensor([5.0]))}")

for epoch in range(n_iter):
    # forward pass
    y_predicted = model(x)
    loss = criterion(y, y_predicted)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    # reset gradientsfor epoch in range(n_iter):
    # forward pass
    y_predicted = model(x)
    loss = criterion(y, y_predicted)            # we get mean loss of batch

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()

    # reset gradients
    optimizer.zero_grad()

    if epoch%1==0:
        (w,b) = model.parameters()
        print(f"epoch {epoch}, loss {loss}, w {w[0][0].item()}, b {b.item()}")


print(f"After training, f(5) = {model(torch.tensor([5.0]))}")