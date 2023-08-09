# Learning Transferable Visual Models From Natural Language Supervision

## Idea:

At the core of clip's approach is the idea of learning perception from supervision contained in natural language. This also will enable leveraging larger datasets as image-text pairs can be scraped from the web instead of using limiting supervision of a few thousand labels of classification datasets(Imagenet)

If we somehow learn to connect image to the text that comes along with it, we wouldnt be bound by the labels, and we can learn very good representations of images. 
If we have such a model, we can use the model representations to finetune it to other tasks. We can use zero-shot imagenet accuracy as a proxy to compare the representations we get using different pretraining approaches.


<br>

## Zero shot classification

Q. How to go from a model that predicts text to a zero-shot classifier?  
-> If we have a model that can predict text (ie return probability scores) given an image, we can use these prob scores to compare (cosine similarity) with any labels (dog, cat, none). the one with the max score is the predicted label. this way, a feature representer of an image that somehow gives values that map to words can be used as zero-shot classifiers to classify among any unseen labels too.

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

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1U22s5Z6sOhOeYhQl5aB5rhTbiKNVxK4l" alt="contrastive learning in clip"/>
</p>

*Minibatches must be of large size since for each minibatch, this $I*T$ matrix is created and in each the diagonal values are maximized and other values minimized. Hence only info from that batch serves as supervision.*

<br>

## Encoders

TEXT:  

Transformer acrhitecture. As a base size we use a 63M-parameter 12-layer 512-wide model with 8 attention heads. The transformer operates on a lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size For computational efficiency, the max sequence length was capped at 76. The text sequence is bracketed with [SOS] and [EOS] tokens and the **activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text** which is layer normalized and then linearly projected into the multi-modal embedding space.

IMAGE:

Resnets, VITs




## Results

- The results are cometitive with fully supervised linear classifier Resnet-50 on 16 datasets including Imagenet.

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

- linear probing: keeping the bacbone fixed and training a small classifier head on top. (opposed to finetuning where backbone weights are also finetuned)
- zero shot classification
- contrastive learning
- few shot learning


Q how is few shot performed?
