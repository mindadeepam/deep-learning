#   1. prepare data
#   2. define model
#   3. define loss and optim
#   4. define training loop-
#       - forward pass
#       - backward pass
#       - update weights, reset gradients

import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


# 1. prepare data
bc = datasets.load_breast_cancer()
X, y = bc['data'], bc['target']
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
std_sc = StandardScaler()                                               # feature scaling
X_train = std_sc.fit_transform(X_train)
X_test = std_sc.transform(X_test)

X_train = torch.tensor(X_train.astype(np.float32))
X_test = torch.tensor(X_test.astype(np.float32))
y_train = torch.tensor(y_train.astype(np.float32)).view(y_train.shape[0],-1)
y_test = torch.tensor(y_test.astype(np.float32)).view(y_test.shape[0],-1)


n_samples, n_features = X.shape
input_shape = n_features
output_shape = 1


# 2. define model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear=nn.Linear(input_dim, 1, dtype=torch.float32)     # default dtype here seems to be float32, if ur inputs are diff, specify the same dtype here

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
        # return self.linear(x)                 # use with nn.BCEWithLogitsLoss()


# 3. define loss, optim and other training parameters
model = LogisticRegression(input_shape, output_shape)
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
n_iter = 100

# model(X_train)

# 4. training loop
for epoch in range(n_iter):
    # forward pass
    output = model(X_train)
    loss = criterion(output, y_train)

    # backward pass
    loss.backward()

    # update weights
    optimizer.step()
    optimizer.zero_grad()

    if epoch%10==0 or epoch==n_iter-1:
        print(f"epoch {epoch+1}, loss {loss}")

with torch.no_grad():
    output = model(X_test)
    y_pred = output>0.5                 # output.round() does the same
    test_acc = (torch.sum(y_pred == y_test)).item()/y_test.shape[0]                     # ypred.eq(y_train).sum() /float(test.shape[0])
    print(f'Test Accuracy {test_acc}, test loss {criterion(output, y_test)}')
######3 why are lossses 00