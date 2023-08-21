# Learning Transferable Visual Models From Natural Language Supervision

## Idea:

At the core of clip's approach is the idea of learning perception from supervision contained in natural language. Learning from natural language has several potential strengths over other training methods. It’s much easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic “machine learning compatible format” such as the canonical 1-of-N majority vote “gold label”. Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet. Learning from natural language also has an important advantage over most
unsupervised or self-supervised learning approaches in that it doesn’t “just” learn a representation but also connects that representation to language which enables flexible zero-shot transfer.

 
The authors demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks.



<br>

## Zero shot classification

Q. How to go from a model that predicts text to a zero-shot classifier?  
-> If we have a model that can predict text, ie return probability scores given an image, we can use these prob scores to compare (cosine similarity) with any labels (dog, cat, none). the one with the max score is the predicted label. this way, a feature representer of an image that somehow gives values that map to words can be used as zero-shot classifiers to classify among any unseen labels too.

(Since the labels we give during zero shot are like prompts, this invites prompt engineering!)  
<br>

## Pretraining

- They initially trained a CNN and text transformer from scratch to predict the caption of images. but this didnt scale and was less efficient than a baseline that predicts bag-of-words rather than the exact caption.  
- Inspired by Recent work in contrastive representation learning for images they switched to predicting only which text as a whole is paired with which image and not the exact words of that text.  

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1PHMnJzi9_oFZQl63jWXjd2quhdQUxiaV" alt="contrastive pretraining approaches"/>
</p>

*CLIP is much more efficient at zero-shot transfer than our image caption baseline. Although highly expressive, transformer-based language models are relatively weak at zero-shot ImageNet classification. Here, we see that it learns 3x slower than a baseline which predicts a bag-of-words (BoW) encoding of the text (Joulin et al., 2016). Swapping the prediction objective for the contrastive objective of CLIP further improves efficiency another 4x.*  

<br>

## Contrastive learning:  

Assume we have a batch of N images paired with their respective descriptions e.g. <image1, text1>, <image2, text2>, <imageN, textN>. 
Contrastive Pre-training aims to jointly train an Image and a Text Encoder that produce image embeddings [I1, I2 … IN] and text embeddings [T1, T2 … TN], in a way that:  
    - The cosine similarities of the correct <image-text> embedding pairs <I1,T1>, <I2,T2> (where i=j) are maximized.  
    - In a contrastive fashion, the cosine similarities of dissimilar pairs <I1,T2>, <I1,T3>… <Ii,Tj> (where i≠j) are minimized.  
They optimize a symmetric cross entropy loss over these similarity
scores.

*pseudocode for CLIP:*
```
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality

I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```


<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1U22s5Z6sOhOeYhQl5aB5rhTbiKNVxK4l" alt="contrastive learning in clip"/>
</p>

*Minibatches must be of large size since for each minibatch, this $I*T$ matrix is created and in each the diagonal values are maximized and other values minimized. Hence only info from that batch serves as supervision.*

<br>

## Encoders

TEXT:  

Transformer acrhitecture. As a base size we use a 63M-parameter 12-layer 512-wide model with 8 attention heads. The transformer operates on a lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size For computational efficiency, the max sequence length was capped at 76. The text sequence is bracketed with [SOS] and [EOS] tokens and the **activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text** which is layer normalized and then linearly projected into the multi-modal embedding space. Masked self-attention
was used in the text encoder.

IMAGE:

Resnets, VITs


## Using CLIP for zero-shot classification
we first compute the feature embedding of the image and the feature embedding of the set of possible texts by their respective encoders.
The cosine similarity of these embeddings is then calculated,
scaled by a temperature parameter τ , and normalized into a
probability distribution via a softmax.

When interpreted this way, the image
encoder is the computer vision backbone which computes a
feature representation for the image and the text encoder is a
hypernetwork (Ha et al., 2016) which generates the weights
of a linear classifier based on the text specifying the visual
concepts that the classes represent



## Results

- The results are cometitive with fully supervised linear classifier Resnet-50 on 16 datasets including Imagenet. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training.

    <p align="center">
        <img src="https://drive.google.com/uc?export=view&id=1POe582Aifb_2tmYiVHShhvTKpa1rO7a6" alt="zero-shot lip vs fully trained resnet"/>
    </p>
    
    *curiously it performs very poorly on simple datasets like MNIST. It would be fair to say it wirks well on images that are generally available on the internet because it was trained on such data. for digits/documents/etc which it wasnt trained in it is expected to perform poorly.*

- prompt engineering and ensembling classes helps performance. This could include more detailed prompts, more adverserial classes or ensembling classes 

    <p align="center">
        <img src="https://drive.google.com/uc?export=view&id=173V06XfYoJaENJMIyolMtW-ceQ8ySbiB" alt="prompt engineering and ensemblinfg"/>
    </p>
    
<br>

---
## References
- [paper](https://arxiv.org/pdf/2103.00020.pdf)
- Intuitive explanation of core concepts in CLIP. [Medium article](https://towardsdatascience.com/clip-the-most-influential-ai-model-from-openai-and-how-to-use-it-f8ee408958b1)



## new terms

- linear probing: keeping the backbone fixed and training a small classifier head on top. (opposed to finetuning where backbone weights are also finetuned). For linear probing, the authors used only the CLIP’s Image Encoder to get the image features and fed them into a linear classifier.
- zero shot classification
- contrastive learning
- few shot learning


Q how is few shot performed?
