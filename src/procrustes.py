import argparse
import copy
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
import wandb
import pandas as pd

from src.config import ModelConfig, MODEL_DIM, MuseConfig
from MUSE.src.evaluation import Evaluator
from MUSE.src.models import build_model
from MUSE.src.trainer import Trainer
from MUSE.src.utils import bool_flag, initialize_exp

class MuseExp:
    def __init__(self, config):
        self.config = config
        
    def run(self, extend_exp=None, model_info=MODEL_DIM, data_split="1k_only"):
        exp_flag = "" if extend_exp is None else f"_{extend_exp}"
        bins = {
            "_freq": ["freq_500", "freq_5000", "freq_end"],
            "_poly": ["1_meaning", "over_3", "2_or_3"],
            "_disp": ["disp0.1", "disp0.5", "disp0.9"]
        }.get(exp_flag[:5], ["original"])

        project_name = f"image2{self.model_type}-TACL-{exp_flag}"
        wandb.init(project=project_name, name=f"{self.src_lang}_{self.tgt_lang}")
        metrics_df = pd.DataFrame()
        # for v_model in model_info["VM"]:
        # assume the input is VM
        for l_model in model_info["LM"]:
            emb_dim = min(model_info["VM"][self.config.model.model_name], model_info["LM"][l_model])
            for bin_name in bins:
                for fold in [203, 255, 633, 813, 881]:
                    metrics = {"VM": self.config.model.model_name,
                                "LM": l_model,
                                "dim": emb_dim,
                                "Bins": bin_name,
                                "Fold": f"fold_{fold}"}
                    muse_params = MuseConfig("imagenet", l_model, self.config.model.model_name, 
                                                emb_dim, fold, bin_name, data_split)
                    muse_res = muse_supervised(muse_params)
                    metrics.update(muse_res)
                    # build dataframe from dictionary
                    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])])
            # metrics.clear()

        wandb.log({"Results": wandb.Table(dataframe=metrics_df.round(2))}, commit=True)
        wandb.finish()


def muse_supervised(configs):
    params = argparse.Namespace(**configs)
    # check parameters
    assert not params.cuda or torch.cuda.is_available()
    assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
    assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
    assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
    assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)

    trainer.load_training_dico(params.dico_train)

    VALIDATION_METRIC_SUP = 'precision_at_1-csls_knn_100'
    VALIDATION_METRIC_UNSUP = 'mean_cosine-csls_knn_100-S2T-10000'

    # define the validation metric
    VALIDATION_METRIC = VALIDATION_METRIC_UNSUP if params.dico_train == 'identical_char' else VALIDATION_METRIC_SUP
    logger.info("Validation metric: %s" % VALIDATION_METRIC)

    """
    Learning loop for Procrustes Iterative Learning
    """

    n_iter = 0
    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)
    result_metrics = {
        'P@1-CSLS': to_log["precision_at_1-csls_knn_100"],
        'P@5-CSLS': to_log["precision_at_5-csls_knn_100"],
        'P@10-CSLS': to_log["precision_at_10-csls_knn_100"],
        'P@30-CSLS': to_log["precision_at_30-csls_knn_100"],
        'P@50-CSLS': to_log["precision_at_50-csls_knn_100"],
        'P@100-CSLS': to_log["precision_at_100-csls_knn_100"],
        'P@1-NN': to_log["precision_at_1-nn"],
        'P@5-NN': to_log["precision_at_5-nn"],
        'P@10-NN': to_log["precision_at_10-nn"],
        'P@30-NN': to_log["precision_at_30-nn"],
        'P@50-NN': to_log["precision_at_50-nn"],
        'P@100-NN': to_log["precision_at_100-nn"],
        "mean_cosin": to_log["mean_cosine-csls_knn_100-S2T-10000"]
    }

    return result_metrics


if __name__ == "__main__":
    # main
    parser = argparse.ArgumentParser(description='Supervised training')
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="", help="Export embeddings after training (txt / pth)")

    # data
    parser.add_argument("--src_lang", type=str, default='en', help="Source language")
    parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
    # training refinement
    parser.add_argument("--n_refinement", type=int, default=5,
                        help="Number of refinement iterations (0 to disable the refinement procedure)")
    # dictionary creation parameters (for refinement)
    parser.add_argument("--dico_train", type=str, default="default",
                        help="Path to training dictionary (default: use identical character strings)")
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_100',
                        help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
    parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0,
                        help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    # reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
    parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    parser.add_argument("--load_optim", type=bool_flag, default=False, help="Reload optimal")

    # parse parameters
    config = parser.parse_args()
    muse_supervised(config)

