import torch

x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor([1.0], requires_grad=True)

# forward pass
y_hat = w*x                     # d(y_hat)/d(w) = x
loss = (y_hat-y)**2             # d(loss)/d(y_hat) = 2*(y_hat-y)

print(loss)

loss.backward()         
print(w.grad)

# backward pass:
''' 
w.grad = dloss/dw = dloss/dy_hat * dy_hat/dw
                  = 2*(y_hat-y) *  x
                  = 2*(w*x-y) * x
                  = 2*-1*1
                  = -2
'''
# update weights 
'''optim.step()   '''       

# reset gradients
w.grad.zero_()