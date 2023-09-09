import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from pathlib import Path


def enforce_reproducibility(seed=42):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


# taken from https://github.com/mtoneva/brain_language_nlp/blob/master/utils/utils.py
def CV_ind(n, n_folds):
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))
    for i in range(0, n_folds - 1):
        ind[i * n_items:(i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items:] = (n_folds - 1)
    return ind

def reduce_dim(config, model_info):
    data = torch.load(config.data.alias_emb_dir + f"/{config.model.model_type}/{config.model.model_name}_{config.model.dim}.pth")
    embeddings = data["vectors"]
    for model_type in model_info:
        if model_type == config.model.model_type:
            continue
        for model_name in model_info[model_type]:
            if model_info[model_type][model_name] < int(config.model.dim):
                emb_dim = model_info[model_type][model_name]
                if not Path(config.data.alias_emb_dir + f"/{config.model.model_type}/{config.model.model_name}_{emb_dim}.pth").exists():
                    pca = PCA(n_components=emb_dim, random_state=config.seed)
                    reduced_emb = pca.fit_transform(embeddings)
                    torch.save({"dico": data["dico"], "vectors": torch.from_numpy(reduced_emb).float()},
                                config.data.alias_emb_dir + f"/{config.model.model_type}/{config.model.model_name}_{emb_dim}.pth")
                    print(f"Saved {config.model.model_name}_{emb_dim}.pth")
                    
                    del pca, reduced_emb