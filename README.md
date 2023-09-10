# Introduction

This repository contains code for exploring isomorphism between pre-trained language and vision embedding spaces. Implement transformer-based language models ([BERT](https://arxiv.org/abs/1810.04805), [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [OPT](https://arxiv.org/abs/2205.01068) and [Llama 2](https://arxiv.org/abs/2307.09288)) to get words embeddings, and implemennt Vision Models ([ResNet](https://arxiv.org/abs/1512.03385), [Segformer](https://arxiv.org/abs/2105.15203), and [MAE](https://arxiv.org/abs/2111.06377)) to get images embeddings.

Before running the code, please download fasttext identification [model](https://fasttext.cc/docs/en/language-identification.html)

## Get word embeddings

To get the embeddings of specific words in the wordlist, simply run:
```bash
python main.py --pretrained meta-llama/Llama-2-7b-hf --model_type LM 
```
## Get image embeddings
To get the embeddings of specific image class, simply run:
```bash
python3 main.py --pretrained facebook/vit-mae-huge --model_type VM
```
## Align word and image embeddings
To learn a mapping between the source and the target space, simply run:
```bash
python3 main.py --pretrained meta-llama/Llama-2-7b-hf --model_type LM --muse True
```