import argparse

seed = 42
dataset_name = "imagenet"
sentences_path = "./data/sentences_lv.json"
wordlist_path = "./data/labels.json"
alias_emb_dir = "./data/exps/embeddings" # path to save word embeddings (decontextualized)
emb_per_object = True
num_classes = 1000000 # number of sample ratio
image_dir = "/projects/nlp/people/kfb818/Dir/datasets/imagenet_21k_small"
dictionary_path = "./data/dicts"

MODEL_DIM = { 
    "VM": {
        'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048, 'resnet152': 2048,
        "segformer-b0-finetuned-ade-512-512":256,
        "segformer-b1-finetuned-ade-512-512":512,
        "segformer-b2-finetuned-ade-512-512":512,
        "segformer-b3-finetuned-ade-512-512":512,
        "segformer-b4-finetuned-ade-512-512":512,
        "segformer-b5-finetuned-ade-640-640":512,
        "vit-mae-base":768, "vit-mae-large":1024, "vit-mae-huge":1280},
    "LM": {
        'bert_uncased_L-2_H-128_A-2': 128, 'bert_uncased_L-4_H-256_A-4': 256, 'bert_uncased_L-4_H-512_A-8': 512,
        'bert_uncased_L-8_H-512_A-8': 512, 'bert-base-uncased': 768, 'bert-large-uncased': 1024,
        'gpt2': 768, 'gpt2-medium': 1024, 'gpt2-large': 1280, 'gpt2-xl': 1600,
        'opt-125m':768, 'opt-1.3b':2048, 'opt-6.7b':4096, 'opt-30b':7168,
        "Llama-2-7b-hf": 4096, "Llama-2-13b-hf":5120}
    }


class ModelConfig(argparse.Namespace):
    def __init__(self, model_type="LM", pretrained_model='bert-base-uncased', muse=False):
        super().__init__()
        self.seed = seed
        self.model = argparse.Namespace(**{
            "model_type": model_type,
            "pretrained_model": pretrained_model,
            "model_name": pretrained_model.split('/')[-1],
            "dim": MODEL_DIM[model_type][pretrained_model.split('/')[-1]]})

        self.data = argparse.Namespace(**{
            "dataset_name": dataset_name,
            "sentences_path": sentences_path,
            "wordlist_path": wordlist_path,
            "alias_emb_dir": alias_emb_dir,
            "emb_per_object": emb_per_object,
            "num_classes": num_classes,
            "image_dir": image_dir
        })


# f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_{layer}_per_sentence.pth")
class MuseConfig(argparse.Namespace):
    def __init__(self, dataset, lm, vm, dim, fold, bin_name, data_range ) -> None:
        super().__init__()
        self.muse_params = argparse.Namespace(**{
            "tgt_lang": "lang",
            "src_lang": "image",
            "n_refinement": 0,
            "dico_eval": dictionary_path + f"/test/test_{fold}{bin_name}_{data_range}.txt",
            "dico_train": dictionary_path + f"/train/train_{fold}{bin_name}_{data_range}.txt",
            "src_emb": f"/VM/{dataset}_{vm}_dim_{dim}_layer_last.pth",
            "tgt_emb": f"/LM/{dataset}_{lm}_dim_{dim}_layer_last.pth",
            "exp_name": "./exps/muse_results",
            "exp_id": "",
            "cuda": True,
            "export": "",
            "emb_dim": dim,
            "max_vocab": 200000,
            "dico_method": "csls_knn_100",
            "dico_build": "S2T&T2S",
            "load_optim": False
        })
        
        

