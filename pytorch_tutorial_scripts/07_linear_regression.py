# 1. define model
# 2. define loss, optimizers
# 3. define training loop - 
#       1. forward pass
#       2. backward pass
#       3. update weights


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets 
import matplotlib.pyplot as plt

X_np,y_np = datasets.make_regression(n_samples=500, n_features=1, noise=20, random_state=42)            #same code works with 20 features too, only plotting becomes tough

X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0], 1)

# input output shapes are always regarding num of features in each sample
n_samples, n_features = X.shape
input_shape = n_features
output_shape = 1
model = nn.Linear(input_shape,output_shape)

# class LinearRegression(nn.module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(LinearRegression,self).__init__()

#         self.lin1 = nn.Linear(input_dim, hidden_dim)
#         self.lin2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         output = nn.Tanh(self.lin1(x))

lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
n_iter = 100

for epoch in range(n_iter):

    y_predicted = model(X)                  # all inputs at once 
    loss = criterion(y, y_predicted)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch%10==0 or epoch==n_iter-1:
        w, b = model.parameters()
        print(f"epoch {epoch}, loss {loss:.3f}")


# plot results
predictions = model(X).detach().numpy()
plt.plot(X_np, y_np, 'go')
plt.plot(X_np, predictions, 'r')
plt.show()
