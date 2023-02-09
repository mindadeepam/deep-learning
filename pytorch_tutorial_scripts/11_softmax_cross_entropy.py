import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)           #keepdims helps broadcast shape across all columns ineach sample

logits = np.array([[0.000001, 1.0, 0.1], [1.0, 10.0, 2000.0]])
y1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],)           # one-hot
y2 = np.array([0,1])        # labels


outputs=softmax(logits)
print(outputs)
print(torch.softmax(torch.from_numpy(logits), dim=1))


def NLLLoss(outputs, y):
    ce = -1*np.sum(y*np.log(outputs), axis=1)
    print(ce)
    return np.mean(ce,keepdims=True)

loss=nn.CrossEntropyLoss()
print(loss(torch.from_numpy(logits), torch.from_numpy(y1)))
print(NLLLoss(outputs, y1))

# how to get prediction labels from logits
_, predictions1 = torch.max(torch.from_numpy(logits), axis=1)
predictions1_np = np.argmax(y1, axis=1)