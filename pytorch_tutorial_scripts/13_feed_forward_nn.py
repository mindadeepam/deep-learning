
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 32
input_size = 28*28
hidden_size = 100
num_classes = 10


# load data
train_dataset = torchvision.datasets.MNIST(root='./tutorial/data/', train=True,
                transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./tutorial/data/', train=False,
                transform=transforms.ToTensor(), download=False)

trainloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)


examples = iter(trainloader)
samples, labels = next(examples)
print(samples.shape, labels.shape)


# visualize data
for i in range(6):
    plt.subplot(2,3, 1+i)
    plt.imshow(samples[i][0], cmap='gray')  
plt.show()


# define model
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = self.linear2(out)
        return out

model = FFNN(input_size, hidden_size)
model.to(device)
                # model(samples.view(batch_size, -1).shape)

# define loss & optim
lr = 0.001
n_iter=10
criterion = nn.CrossEntropyLoss()                # output=logits and y_true=labels/probabilties
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
total_steps = len(trainloader)

best_loss = float('inf')
for epoch in range(n_iter):
    
    # training loop
    total_loss = 0
    print(f"epoch {epoch+1}/{n_iter}")
    for i, batch in enumerate(trainloader):
        inputs, labels = batch
        # flatten the image for linear layer
        inputs, labels = inputs.view(-1, 28*28).to(device), labels.to(device)               # reshape (batchsize, 1, 28, 28) to (-1, 28*28)
        
        #  forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)           # we get mean loss of batch
        
        # backward pass
        loss.backward()        
        # update weights and reset gradients
        optimizer.step()
        optimizer.zero_grad()

        if i%500==0: 
            print(f"\t epoch {epoch+1}/{n_iter}, step {i+1}/{total_steps},  loss {loss}") 
        total_loss = total_loss+loss

    print(f"train_loss {total_loss/total_steps}")


    # eval loop
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        total_eval_loss = 0
        for i, batch in enumerate(testloader):
            inputs, labels = batch
            inputs, labels = inputs.view(-1, 784).to(device), labels.to(device)
            outputs = model(inputs)
            _, y_pred = torch.max(outputs, axis=1)
            correct = y_pred.eq(labels).sum().item()
            n_correct += correct 
            n_samples += labels.shape[0] 
            loss = criterion(outputs, labels)
            total_eval_loss += loss


        test_accuracy = n_correct/n_samples
        test_loss = total_eval_loss / len(testloader)
        print(f"test_accuracy {test_accuracy:.3f} \ntest_loss {test_loss:.3f}")

        if test_loss<best_loss:
            best_loss=test_loss
            print(f"New best loss {best_loss:.3f}")
        else:
            print("test performance getting worse!! stopping training now!!!!")
            break
        
