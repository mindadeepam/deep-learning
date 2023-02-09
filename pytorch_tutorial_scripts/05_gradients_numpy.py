import numpy as np

# Approximate a function: y = f(x) = 2*x
x=np.array([1,2,3,4], dtype=np.float16)
y=np.array([2,4,6,8], dtype=np.float64)

w=0.0

# model prediction
def forward(x):
    return w*(x)

# loss = MSE
def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
def gradient(x, y, y_predicted):
    return np.mean(2*(y_predicted-y)*x)          # * is the same as dot product


print(f"Prediction before trainng: f(5) = {forward(5):.3f}")


lr = 0.01
n_iters=20

for epoch in range(n_iters):
    # forward pass
    y_predicted = forward(x)
    
    # loss
    l = loss(y, y_predicted)
    
    # gradients
    dw = gradient(x, y, y_predicted)
    
    # update weights 
    w -= lr*dw
    
    if epoch%1==0:
        print(f"epoch {epoch+1}, w = {w}, loss = {l:.3f}, prediction during training: f(5) = {forward(5):.3f}")

print(f"\nPrediction after trainng: f(5) = {forward(5):.3f}")
