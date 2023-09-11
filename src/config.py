import argparse

seed = 42
dataset_name = "imagenet"
sentences_path = "./data/sentences.json"
wordlist_path = "./data/all_words.json"
alias_emb_dir = "/projects/nlp/people/kfb818/Dir/datasets/" # path to save word embeddings (decontextualized)
emb_per_object = True
num_classes = 1000000 # number of sample ratio
image_dir = "/projects/nlp/people/kfb818/Dir/datasets/imagenet_21k_small"
dictionary_path = "./data/dicts"
image_id_pairs = "./data/id_pairs_21k.json"

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
        'gpt2': 768, 'gpt2-large': 1280, 'gpt2-xl': 1600,
        'opt-125m':768, 'opt-6.7b':4096, 'opt-30b':7168, 'opt-66b':9126,
        "Llama-2-7b-hf": 4096, "Llama-2-13b-hf":5120, "Llama-2-70b-hf":8192}
        # "LM": {
        # 'bert_uncased_L-2_H-128_A-2': 128, 'bert_uncased_L-4_H-256_A-4': 256, 'bert_uncased_L-4_H-512_A-8': 512,
        # 'bert_uncased_L-8_H-512_A-8': 512, 'bert-base-uncased': 768, 'bert-large-uncased': 1024,
        # 'gpt2': 768, 'gpt2-large': 1280, 'gpt2-xl': 1600,
        # 'opt-125m':768, 'opt-6.7b':4096, 'opt-30b':7168,
        # "Llama-2-7b-hf": 4096, "Llama-2-13b-hf":5120}
    }


class ModelConfig(argparse.Namespace):
    def __init__(self, model_type="LM", pretrained_model='bert-base-uncased'):
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
            "image_dir": image_dir,
            "image_id_pairs": image_id_pairs
        })


class MuseConfig(argparse.Namespace):
    def __init__(self, more_exp, lm, vm, dim, fold, bin_name, data_range="cleaned") -> None:
        super().__init__()
        disp_flag = False if len(more_exp)==0 else True
        if disp_flag:
            if "image" in more_exp:
                test_dict_dir = f"{vm}_disp"
            else:
                test_dict_dir = f"{lm}_disp"
        else:
            test_dict_dir = f"original"
        self.hyperparams = argparse.Namespace(**{
            "disp_flag": disp_flag,
            "seed": seed,
            "tgt_lang": lm,
            "src_lang": vm,
            "n_refinement": 0,
            "normalize_embeddings": "center",
            "emb_dim": dim,
            "dico_eval": dictionary_path + f"/{test_dict_dir}/test_{fold}_{data_range}{bin_name}.txt",
            "dico_train": dictionary_path + f"/original/train_{fold}_{data_range}.txt",
            "src_emb": f"/projects/nlp/people/kfb818/Dir/datasets/VM/{vm}_{dim}.pth",
            "tgt_emb": f"/projects/nlp/people/kfb818/Dir/datasets/LM/{lm}_{dim}.pth",
            "exp_name": "./exps/muse_results",
            "cuda": True,
            "export": "",
            # No need to change the following parameters
            "exp_id": "",
            "max_vocab": 200000,
            "dico_method": "csls_knn_100",
            "dico_build": "S2T&T2S",
            "load_optim": False,
            "verbose": 2,
            "exp_path": "",
            "dico_threshold": 0,
            "dico_max_rank": 10000,
            "dico_min_size": 0,
            "dico_max_size": 0,
        })
        
        

