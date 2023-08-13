# CNN Architectures


## Spatial Dropout and BatchNorm 

[This](https://youtu.be/TGqqTgn4cAg) is the youtube link.

Spatial Dropout

 - Problem with regular dropout is that in images adjacent pixels are highly correlated. so even if we drop random pixels in a recptive field,
 - Hence we drop entire channels. (dropout in channel dimension)

[Batchnorm paper (2015)](https://arxiv.org/pdf/1502.03167.pdf)
at its time improved SOTA on Imagenet reaching 4.8% top5 test error.

- normalizes inputs for each layer. 
- mitigates internal-covariate-shift, ie changing changing distributuion of parameters.
- results in faster convergence (although each epoch takes longer)

- Batch norm smoothens the loss landscape. we can have larger learning rates. graidents are smoother.

Heres an [article](https://ketanhdoshi.github.io/Batch-Norm-Why/) that has useful visualizations and intuition. 
<div style="display: flex;">
    <img src="https://drive.google.com/uc?export=view&id=1cCe-2OYeqCHGhautGuxjuNeFWD5lu2eo" alt="Image 1" width="50%" style="margin-right: 10px" />
    <img src="https://drive.google.com/uc?export=view&id=1C3Ms64FeLHk2e4iaHkDy-T5ahdva37Xd" alt="Image 2" width="50%" style="margin-left: 10px"/>
</div>    
<br>

It first normalizes (subtracts mean divide by std) then it scales and shifts using learnable params $\gamma\ and\ \beta$. For images ie 4d inputs, there are as many $\gamma s\ and\ \beta s$ as there are number of channels.

```python
bn_layer = torch.nn.BatchNorm2d(num_features = num_output_channels_of_prev_layer)
```
*Refer to this [doc](../../awesome-papers/batch-norm.md) for more nuanced understanding.*




## Many CNN architectures
There are a plethora of CNN architectures available out there.

have a look at these references:
- [A Survey of the Recent Architectures of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1901.06032.pdf) recommended
- [Review of deep learning: Concepts, CNN architectures, challenges, applications, future directions](https://link.springer.com/article/10.1186/s40537-021-00444-8)
- [Backbones-Review: Feature Extraction Networks for Deep Learning and Deep Reinforcement Learning Approaches](https://arxiv.org/pdf/2206.08016.pdf)
- [A Survey of Convolutional Neural Networks:
Analysis, Applications, and Prospects](https://arxiv.org/pdf/2004.02806.pdf)
