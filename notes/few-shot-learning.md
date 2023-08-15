# Few shot learning

## TEXT

There are a few methods we could use for few shot learning in text. 

- **In conext learning**, aka describing tasks to LLMs using prompts. No weights are updated in this approach.  
While this sounds like an attractive option, given all the LLM hype these days, performance of these models increases with scale. It is costly to deploy LLMs.  
Training or even inference using these models is computationally expensive. A 7B model occupies almost 28Gb GPU.
(7B params * 4bytes/param for float32 = 28Gb GPU. refer [this](https://huggingface.co/docs/transformers/perf_train_gpu_one#anatomy-of-models-memory))  

- An additional paradigm for enabling a model to perform a new task with minimal updates is **parameter efficient fine-tuning (PEFT)**, where a pre-trained model is fine-tuned by only updating a small number of added or selected parameters.  
PEFT methods include LoRA, Prefix-tuning, P-tuning, prompt tuning, AdaLoRA, $(IA)^3$. see Appendix 2.  
[**T-Few**](https://arxiv.org/pdf/2205.05638.pdf) is current SOTA. In practice,
    - performance sensitive to quality of "prompt engineering" (like ICL)
    - SOTA performance relies on largish 11B (11.4GB) T0 model


<br>

- another paradigm is "pattern exploiting training", ie to train using prompts (cloze questions) aka **PET and adaPET**. But in practise,  
    - requires hand-crafted "prompt templates" and "verbalizers"
    - slow to train (variable length verbalizers) and run inference (autoregressive decoding)  

<br>

- Still another approach is to use embeddings from PLMs and use a metric based approach to few-shot learning. This builds on other metric based approaches like matching, prototype, relation networks, [etc](https://lilianweng.github.io/posts/2018-11-30-meta-learning/#metric-based).  
This method has numerous implementations, 
    - using pretrained embeddings for zero shot classification [link](https://joeddav.github.io/blog/2020/05/29/ZSL.html)
    - using contrastive learning, as in [setfit](https://arxiv.org/pdf/2209.11055.pdf)




### References

1. Induction Networks for Few-Shot Text Classification [2019] [[paper](https://arxiv.org/pdf/1902.10482v2.pdf)]
2. Efficient Few-Shot Learning Without Prompts aka **Setfit** [2019] [[paper](https://arxiv.org/pdf/2209.11055.pdf)]
3. Few-Shot Parameter-Efficient Fine-Tuning is Better
and Cheaper than In-Context Learning aka T-few [2022] [[paper](https://arxiv.org/pdf/2205.05638.pdf)]
4. Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference aka PET [2020] [[paper]](https://arxiv.org/pdf/2001.07676.pdf)
5. Cloudera few shot report [2020] [[article]](https://few-shot-text-classification.fastforwardlabs.com/)




others

- Lifelong domain word embedding via meta-learning [2018] [[paper](https://arxiv.org/pdf/1805.09991.pdf)]






## Appendix

#### Huggingface's zero-shot pipeline uses NLI models.   
    NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification` trained on NLI (natural
    language inference) tasks. Equivalent of `text-classification` pipelines, but these models don't require a
    hardcoded number of potential classes, they can be chosen at runtime. It usually means it's slower but it is
    **much** more flexible.

    Any combination of sequences and labels can be passed and each combination will be posed as a premise/hypothesis pair and passed to the pretrained model. Then, the logit for *entailment* is taken as the logit for the candidate label being valid. Any NLI model can be used, but the id of the *entailment* label must be included in the model config's :attr:*~transformers.PretrainedConfig.label2id*.

#### PEFT methods:  
    - LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
    - Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
    - P-Tuning: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
    - Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
    - AdaLoRA: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)  
    - $(IA)^3$ aka T-few : [Infused Adapter by Inhibiting and Amplifying Inner Activations ](https://arxiv.org/abs/2205.05638)