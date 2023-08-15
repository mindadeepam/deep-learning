# Efficient Few-Shot Learning Without Prompts


[[paper](https://arxiv.org/pdf/2209.11055.pdf)] 
[[video](https://www.youtube.com/watch?v=8h27lV8v8BU)]

Currently there are 3 main approaches to few shot learning - 

### 1. In context learning

- no gradient updates. like in GPT3
- performance increases with scale
- Hence:
    - performance sentence to quality of “prompt engineering”
    - hard/expensive to deploy large models

### 2. PET

- Alternatives include "pattern exploiting training" (PET) and ADAPET which convert inputs to cloze-style format similar to MLM
- In practice:
    - requires hand-crafted "prompt templates" and "verbalizers"
    - slow to train (variable length verbalizers) and run inference (autoregressive decoding)

![Screenshot 2023-08-11 at 7.28.27 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fb35d03e-1051-493f-95ed-e75d97f02c19/Screenshot_2023-08-11_at_7.28.27_PM.png)

### 3. PEFT

- Alternatives include "parameter efficient fine-tuning" (PEFT) which add/update a small number of parameters in a pretrained LM. [**T-Few**](https://arxiv.org/pdf/2205.05638.pdf) is current SOTA.
- In practice:
    - performance sensitive to quality of "prompt engineering" (like ICL)
    - SOTA performance relies on largish 11B (11.4GB) T0 model

![Screenshot 2023-08-11 at 7.30.17 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c84b83e4-63ca-41ae-b344-1421c19fcd3b/Screenshot_2023-08-11_at_7.30.17_PM.png)


## Results
![table](https://drive.google.com/uc?export=view&id=1c4v0y9g3-gTxM_upUJuanMT5_Mz3TOQK)

![raft-results](https://drive.google.com/uc?export=view&id=1bBJqlLfR0SlMHh9tN0K3HWB4WQp1NwHX)

## Multilingual Experiments
![table](https://drive.google.com/uc?export=view&id=17TxxVqMkj5XKEmBi7P60Zd0UWYWVubJZ)


A higher MAE indicates weaker performance. In the few-shot regime of N = 8 samples per class, we find that SETFIT significantly outperforms FINETUNE and ADAPET in all settings (each, en, all), with the best average performance obtained when training on English data only.

Fully finetuned models are still better than set-fit.

*Models used are of similar sizes.

## SETFIT Model Efficiency
- Few-shot distillation: Using teacher-student framework reduces computation cost while maintaining performance.  The SETFIT student significantly outperforms the baseline student when only small amounts of unlabeled data are available. For example, for N = 8, the SETFIT student outperforms the baseline student by 24.8, 25.1, and 8.9 average accuracy on the AGNews, Emotion and SST5 datasets, respectively. As N increases, the performance gains decrease and are on par for N = 1K.