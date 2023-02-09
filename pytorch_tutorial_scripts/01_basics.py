import torch
import numpy as np

torch.cuda.is_available()

torch.zeros(1, dtype=int)
torch.zeros(1, dtype=float)
torch.zeros(2,4,5, dtype=torch.float64)


# shape/size is size in each dimension 
# (outermost, ..--->.. , innermost) ie (.., row, column)
x = torch.empty(2,3, dtype=torch.double)
x.size()
x.shape                             
x.dtype

x = torch.tensor([[1,2,3,4],[3,2,1,43]])

# torch.rand for uniform distribution b/w [0,1)
x = torch.rand(2,4)         
y = torch.rand(2,4)
y.add_(x)                   #inplace addition
z = torch.add(x,y)
z = x+y

# randint for tensor of integers bw low and high of given shape
z = torch.randint(low=1,high=6, size=(3,3,4))
z
# randn for normal distrbution with mean=0 and variance=1
z=torch.randn(3,2)
z

#[rows, columns] or in general (outermost.. , row, column)
z[:,3]  # all rows 3rd colum
z[1]

z = torch.rand(2,3,3, dtype=torch.float)
z[:,:,:]
n = np.random.rand(2,3,3)
n


# .item() to get scalar value of 1 element tensor ie shape=1 or (1,1) or (1,1,1)...
z[1,1,1].item()
x = torch.rand(1,1)
x.item()

# reshape tensors
z = torch.rand(3,4,4)
z 
z.view(-1)          # fit all into 1 row
z.view(12,4)        # 12 rows 4 cols
z.view(2,2,-1)      # any number of cols, 2 rows, 2 "sheets.."


# tensor to numpy  
x=torch.rand(2,3)
x
y=x.numpy()         # if tensor is in cpu, both point to the same values in memory
y 
x.add_(1)           # adds 1 to each element
x, y                # y also get updated

# numpy to tensor
x = np.ones((3,2))
y = torch.from_numpy(x)
y.add_(3)
x, y                # again both are updated (only on CPU)

# check for gpu and use it
device = 'cuda' if torch.cuda.is_available() else "cpu"
device
if device=='cuda':
    x = torch.rand(2,2).to(device)      # torch.rand(1,1, device=device)
    y = x.cpu().numpy()                 # or x.to('cpu')
    x.add_(2)

    print(x, "\n", y)               # different values, np array and tensor no longer point to the same memory


torch.rand(2,1, device=device)

# gradients
x = torch.rand(2,1, requires_grad=True)
x