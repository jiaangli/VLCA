# Do Vision and Language Models Share Concepts? A Vector Space Alignment Study

This is the code to replicate the experiments described in the paper. 

> Jiaang Li, Constanza Fierro, Yova Kementchedjhieva, and Anders Søgaard. ["Do Vision and Language Models Share Concepts? A Vector Space Alignment Study"](https://arxiv.org/abs/2302.06555) arXiv preprint arXiv:2302.06555 (2023).

We implement transformer-based language models ([BERT](https://arxiv.org/abs/1810.04805), [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [OPT](https://arxiv.org/abs/2205.01068), and [Llama 2](https://arxiv.org/abs/2307.09288)) to obtain word embeddings. Additionally, we implement Vision Models ([ResNet](https://arxiv.org/abs/1512.03385), [Segformer](https://arxiv.org/abs/2105.15203), and [MAE](https://arxiv.org/abs/2111.06377)) to obtain image embeddings.

## Setup
You can clone this repository issuing:
```bash
git@github.com:jiaangli/vislm-geo.git
```

1\. Create a fresh conda environment and install all dependencies.
```text
conda create -n vislm-geo python=3.11
conda activate vislm-geo
pip install -r requirements.txt
```
2\. Download the datasets

imagenet-21k and cldi


## How to run

See available model configurations in [`config.py`](./src/config.py) under `MODEL_CONFIGS`, available saving paths of datasets under `DataConfig`, runtime parameters under `MuseConfig`, and various experimental types under `ExperimentType`.

Set the corresponding paths in all the files in [`conf`](./conf) folder.

Example to sequentially run GPT2 and OPT-125m models on ImageNet-21K dataset:

```bash
python main.py \
    --multirun \
    +model=gpt2,opt-125m \
    +dataset=imagenet \
    muse.exp_type=BASE
```

## How to Cite
If you find our code, data or ideas useful in your research, please consider citing the paper:
```bibtex
@article{li2023implications,
  title={Do Vision and Language Models Share Concepts? A Vector Space Alignment Study},
  author={Li, Jiaang and Kementchedjhieva, Yova and S{\o}gaard, Anders},
  journal={arXiv preprint arXiv:2302.06555},
  year={2023}
}
```

## Acknowledgement

Our codebase heavily relies on these excellent repositories:
- [MUSE](https://github.com/facebookresearch/MUSE)
- [transformers](https://github.com/huggingface/transformers)

<!-- ## Get word embeddings
To get the embeddings of specific words in the wordlist, simply run:
```bash
python main.py 
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
``` -->