import gc
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ..config import ModelType


def enforce_reproducibility(seed: int = 42) -> None:
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
def CV_ind(n: int, n_folds: int) -> np.ndarray:
    ind = np.zeros((n))
    n_items = int(np.floor(n / n_folds))
    for i in range(0, n_folds - 1):
        ind[i * n_items : (i + 1) * n_items] = i
    ind[(n_folds - 1) * n_items :] = n_folds - 1
    return ind


def reduce_dim(config: DictConfig, models: dict) -> None:
    if config.model.model_type == ModelType.LM:
        data_path = (
            Path(config.common.alias_emb_dir)
            / config.model.model_type.value
            / f"{config.model.model_name}_{config.model.dim}.pth"
        )
    else:
        data_path = (
            Path(config.common.alias_emb_dir)
            / config.model.model_type.value
            / config.dataset.dataset_name
            / f"{config.model.model_name}_{config.model.dim}.pth"
        )
    data = torch.load(data_path)
    embeddings = data["vectors"]
    for model_name in models:
        model_info = models[model_name]
        if model_info.model_type == config.model.model_type:
            continue

        if model_info.dim >= int(config.model.dim):
            continue

        emb_dim = model_info.dim
        save_path = data_path.parent / f"{config.model.model_name}_{emb_dim}.pth"
        if not save_path.exists():
            pca = PCA(n_components=emb_dim, random_state=config.common.seed)
            reduced_emb = pca.fit_transform(embeddings)
            torch.save(
                {
                    "dico": data["dico"],
                    "vectors": torch.from_numpy(reduced_emb).float(),
                },
                save_path,
            )
            print(f"Saved {config.model.model_name}_{emb_dim}.pth")
            del pca, reduced_emb
            gc.collect()


def normalization(vecs) -> torch.Tensor:
    # return vecs.sub_(vecs.mean(0, keepdim=True).expand_as(vecs))
    mms = MinMaxScaler()
    vecs = torch.from_numpy(mms.fit_transform(vecs.cpu().numpy()))
    # vecs = torch.from_numpy(zscore(vecs.cpu().numpy()))
    # vecs = vecs.sub_(vecs.mean(0, keepdim=True).expand_as(vecs))
    return vecs.float()
    # return vecs
