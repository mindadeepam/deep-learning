
---
TLDR:  
[Batchnorm paper (2015)](https://arxiv.org/pdf/1502.03167.pdf)
at its time improved SOTA on Imagenet reaching 4.8% top5 test error.

- normalizes inputs for each layer. 
- mitigates internal-covariate-shift, ie changing changing distributuion of parameters.
- results in faster convergence (although each epoch takes longer)

- Batch norm smoothens the loss landscape. we can have larger learning rates. graidents are smoother.

[Here]((https://ketanhdoshi.github.io/Batch-Norm-Why/)) is an article that has useful visualizations and intuition. 
<div style="display: flex;">
    <img src="https://drive.google.com/uc?export=view&id=1cCe-2OYeqCHGhautGuxjuNeFWD5lu2eo" alt="Image 1" width="50%" style="margin-right: 10px" />
    <img src="https://drive.google.com/uc?export=view&id=1C3Ms64FeLHk2e4iaHkDy-T5ahdva37Xd" alt="Image 2" width="50%" style="margin-left: 10px"/>
</div>    
<br>

It first normalizes (subtracts mean divide by std) then it scales and shifts using learnable params $\gamma\ and\ \beta$. For images ie 4d inputs, there are as many $\gamma s\ and\ \beta s$ as there are number of channels.

---
[Batchnorm paper (2015)](https://arxiv.org/pdf/1502.03167.pdf)  
Batch Normalization (BatchNorm) is a technique commonly used in deep neural networks to improve training stability and convergence speed. It aims to address the problem of internal covariate shift, where the distribution of inputs to each layer changes during training, making it harder for the network to learn effectively.

The key idea behind BatchNorm is to normalize the inputs of each layer in a mini-batch to have zero mean and unit variance. This is achieved by performing the following steps during training:

1. Compute the mean and variance of the activations in a mini-batch:
$$
   \mu = \frac{1}{m} \sum_{i=1}^{m} x_i
   \sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu)^2
$$
2. Normalize the activations:
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   where $\epsilon$ is a small constant added for numerical stability.

3. Scale and shift the normalized activations:
   $$y_i = \gamma \hat{x}_i + \beta$$
   Here, $\gamma$ is a learnable scaling parameter and $\beta$ is a learnable shift parameter.

During inference, the moving average and variance of the mean and variance are used to normalize the activations.

```
import torch
import torch.nn as nn

class BatchNormalization(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        super(BatchNormalization, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('moving_mean', torch.zeros(num_features))
        self.register_buffer('moving_variance', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            variance = ((x - mean)**2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            self.moving_mean = (1 - self.momentum) * self.moving_mean + self.momentum * mean.squeeze()
            self.moving_variance = (1 - self.momentum) * self.moving_variance + self.momentum * variance.squeeze()
        else:
            mean = self.moving_mean.view(1, -1, 1, 1)
            variance = self.moving_variance.view(1, -1, 1, 1)
        
        normalized_x = (x - mean) / torch.sqrt(variance + self.epsilon)
        scaled_x = self.gamma.view(1, -1, 1, 1) * normalized_x + self.beta.view(1, -1, 1, 1)
        return scaled_x
```
- Example usage in a model
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.batch_norm1 = BatchNormalization(64)
        self.relu = nn.ReLU()
        # ...

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        # ...
        return x
```

- Create an instance of the model  
`model = Net()`

## FAQs
#### Q. Why do we scale and shift after normalization in batch norm?  

In Batch Normalization, after normalizing the inputs, we apply a scaling and shifting operation to the normalized values. This is done to reintroduce the capability for the network to learn and represent different distributions and ranges of activations, while still benefiting from the normalization that helps with faster convergence and improved generalization.

The scaling and shifting are introduced through learnable parameters: γ (gamma) and β (beta). Here's why we use these parameters:

**Scaling (γ): The scaling factor**  
γ allows the network to control the amplitude of the normalized activations. It's a learnable parameter that enables the network to adapt the magnitudes of the normalized values to the specific requirements of the layer. For example, if the layer requires larger activations for better learning, the network can increase the γ value. Conversely, if smaller activations are more suitable, γ can be decreased. This flexibility enhances the expressiveness of the network.

**Shifting (β): The shifting parameter**   
β enables the network to control the mean of the activations after normalization. This is important because normalization can shift the mean of activations towards zero, which might not be ideal in all cases. By applying the shifting operation, the network can restore or adjust the mean activation to better fit the target distribution. This is particularly helpful when the optimal mean of the activations is not zero, as might be the case in certain layers of the network.

In essence, while normalization helps mitigate the internal covariate shift and provides benefits like faster training and regularization, scaling and shifting allow the network to retain the freedom to determine the best distribution of activations for each layer, which can vary based on the architecture, task, and other factors. This combination of normalization, scaling, and shifting contributes to the overall effectiveness of Batch Normalization in improving the training of deep neural networks.




