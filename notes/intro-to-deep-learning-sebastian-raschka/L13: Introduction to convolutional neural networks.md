
# Introduction to CNNs


### Terminology
- depth - num of channels / $C_o$
- filters / kernels / feature detectors - $K$
- activation / feature map - is the result of conv operation by one filter on input image/tensor. 
- receptive field - kernel sized portion of input image that the conv layer acts on


---

While feedforward networks assume independent features, and sequential networks assume sequential relationship, CNNs assume local feature relation.

![image](https://drive.google.com/uc?export=view&id=1mbGw7eCgQJlJjU9i7mM0cPHB62u2MmHK)



## Intuition on the Conv layer
[link](https://cs231n.github.io/convolutional-networks/#case)  
The CONV layer’s parameters consist of a set of learnable filters. Every filter is small spatially (along width and height), but extends through the full depth of the input volume. For example, a typical filter on a first layer of a ConvNet might have size 5x5x3 (i.e. 5 pixels width and height, and 3 because images have depth 3, the color channels). During the forward pass, we slide (more precisely, convolve) each filter across the width and height of the input volume and compute dot products between the entries of the filter and the input at any position. As we slide the filter over the width and height of the input volume we will produce a 2-dimensional activation map that gives the responses of that filter at every spatial position. Intuitively, the network will learn filters that activate when they see some type of visual feature such as an edge of some orientation or a blotch of some color on the first layer, or eventually entire honeycomb or wheel-like patterns on higher layers of the network. Now, we will have an entire set of filters in each CONV layer (e.g. 12 filters), and each of them will produce a separate 2-dimensional activation map. We will stack these activation maps along the depth dimension and produce the output volume.



## Main concepts of CNN 
[link](https://youtu.be/7fWOE-z8YgY?t=825)  
1. sparse connectivity: a single patch in feature map is connected to only a small patch of image (in MLPs there is dense/full connection)
2. parameter sharing: the same kernel/filter slides across the image. ie different neurons in each activation map is calculated using the same filter. In MLPs each neuron in the output space is calculated using different weight values.
3. many layers: combining extracted local patterns to global patterns. 



## All about the shapes

If stride=1, pad=0, etc all are default values, $$ H_o = (H_i - K + 2P)/S + 1 $$ $$H_{o} = H_{i} - K + 1$$
For the 1st hidden layer, size decreases from (32,32) to (28,28). hence kernel is (5,5). Since channels go from 1 to 6, num_of_filters/depth = 6.  
- These filters are learnable parameters (There is one bias for each output channel. Each bias is added to every element in that output channel)
$$W = [C_o, C_i,  K_h, K_w]$$
$$bias = [C_o]$$
- **Pooling layers** are not learnable, they just downsampling operation along the spatial dimensions $(H, W)$ by avg/max/min pooling. There is information lost due to pooling.

Each Conv2d layer is defined by $[C_i, C_o,  K_h, K_w]$,  
ie there are $C_{out}$ filters and each is of size - $$one\ kernel = [C_i, K_h, K_w]$$  
Sicne each kernels' depth = $C_i$, each filter produces one channel in the resulting matrix irrespective of num of input channels.  
Hence if $$Input = [C_i, H_i, W_i] = [3,32,32],$$ $$Conv2d layer = [C_i, C_o, K_h, K_w] = [3, 16, 3, 3], $$ then there are $$n=16\ kernels $$ each of size $$kernel = [C_i, K_h, K_w] = [3, 3, 3]$$ $H_o = 32-3+1 = 29$ and same with $W_o$. So, $$Output = [C_o, H_o, W_o] = [16, 29, 29]$$

see [this](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) pytorch documentation on Conv2D for more nuanced understanding on the shapes.


## Cross-Correlation vs Convolution

pass

## Overview of different architectures

1. LeNet-5 Yann Lecun and his colleagues in the late 1990 was one of the first CNNs. It was trained to recognize hand-written digit in images.
2. breakthrough came in 2012 when for the 1st time, a CNN architecture, **AlexNet**, beat other DL methods on the ImageNet challenge. that too by a hufe margin(~15% top 5 error comapered to ~26% for 2nd best) 


## What a CNN can see

Use different interpretation methods that help visualize which parts if the image the model is focussing on when inferencing. good debugging tool.  
Some of the methods have been explained in [this](https://thegradient.pub/a-visual-history-of-interpretation-for-image-recognition/) article.  
[This](https://pypi.org/project/grad-cam/) is a library that provides most of these methods.


## References
- Course material [here](https://sebastianraschka.com/blog/2021/dl-course.html#l13-introduction-to-convolutional-neural-networks)  
- CS231N CNN article [here](https://cs231n.github.io/convolutional-networks/#case). Great intuition about the shapes, local connectivity, spatial arrangement, and loads of other stuff.

