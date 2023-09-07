import random

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


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


def uniform_config(args):
    muse_update_params = {
        'emb_dim': args.model.dim,
        'tgt_lang': args.model.model_name,
        'dico_train': args.data.dict_dir,
        'dico_eval': args.data.dict_dir,
        'src_emb': args.data.word_level_fmri_rep_dir,
        'tgt_emb': args.data.word_decontextualized_embs_dir if 'ft' != args.model.model_alias else args.data.alias_emb_dir,
        'exp_name': args.expdir.expname,
    }

    args.convert_parameters.vec_dim = args.model.dim
    args.muse_parameters.update(muse_update_params)

    return args


# def normalization(vecs):
#     # return vecs.sub_(vecs.mean(0, keepdim=True).expand_as(vecs))
#     mms = MinMaxScaler()
#     vecs = torch.from_numpy(mms.fit_transform(vecs.cpu().numpy()))
#     # vecs = torch.from_numpy(zscore(vecs.cpu().numpy()))
#     # vecs = vecs.sub_(vecs.mean(0, keepdim=True).expand_as(vecs))
#     return vecs.float()
#     # return vecs