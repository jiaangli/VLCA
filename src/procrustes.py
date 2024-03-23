import os
from collections import OrderedDict

import pandas as pd
import torch
import wandb
from omegaconf import DictConfig

from MUSE.src.evaluation import Evaluator
from MUSE.src.models import build_model
from MUSE.src.trainer import Trainer
from MUSE.src.utils import initialize_exp
from src.config import MODEL_CONFIGS, ExperimentsType, ModelType, MuseConfig


class MuseExp:
    def __init__(self, config: DictConfig):
        self.config: DictConfig = config
        self.seed = config.common.seed
        self.model_type: ModelType = config.model.model_type
        self.model_name: str = config.model.model_id.split("/")[-1]

    def run(
        self,
        exp_type: ExperimentsType = ExperimentsType.BASE,
        model_info: dict = MODEL_CONFIGS,
        data_type: str = "cleaned",
    ) -> None:
        bins = {
            "freq": ["_freq10000", "_freq100000", "_freq_end"],
            "poly": ["_1", "_over_3", "_2_or_3"],
            "lang_disp": ["_low", "_medium", "_high"],
            "image_disp": ["_low", "_medium", "_high"],
        }.get(exp_type.value, [""])

        project_name = f"img2{self.model_type.value}-{data_type}-{exp_type.value}"
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=project_name, name=f"{self.model_name}")
        metrics_df = pd.DataFrame()
        input_model = model_info[self.model_name]  # model in the command line

        for other_model_name in model_info:
            other_model = model_info[other_model_name]
            if self.model_type == other_model.model_type:
                continue
            emb_dim = min(input_model.dim, other_model.dim)
            if self.model_type == ModelType.LM:
                vm_name = other_model.model_name
                lm_name = self.model_name
            else:
                vm_name = self.model_name
                lm_name = other_model.model_name
            for bin_name in bins:
                for fold in [1, 2, 3, 4, 5]:
                    metrics = {
                        "VM": vm_name,
                        "LM": lm_name,
                        "dim": emb_dim,
                        "Bins": bin_name,
                        "Fold": f"fold_{fold}",
                    }
                    muse_params = MuseConfig(
                        seed=self.seed,
                        exp_type=exp_type,
                        lm=lm_name,
                        vm=vm_name,
                        dim=emb_dim,
                        fold=fold,
                        bin_name=bin_name,
                        data_type=data_type,
                    )
                    muse_res = muse_supervised(muse_params)
                    metrics.update(muse_res)
                    # build dataframe from dictionary
                    metrics_df = pd.concat(
                        [metrics_df, pd.DataFrame(metrics, index=[0])]
                    )

        wandb.log({"Results": wandb.Table(dataframe=metrics_df.round(2))}, commit=True)
        wandb.finish()


def muse_supervised(configs: MuseConfig) -> dict:
    # params = argparse.Namespace(**configs)
    params = configs
    # check parameters
    assert not params.cuda or torch.cuda.is_available()
    assert params.dico_train in ["identical_char", "default"] or os.path.isfile(
        params.dico_train
    )
    assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
    assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
    assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
    assert os.path.isfile(params.src_emb)
    assert os.path.isfile(params.tgt_emb)
    assert params.dico_eval == "default" or os.path.isfile(params.dico_eval)
    assert params.export in ["", "txt", "pth"]

    # build logger / model / trainer / evaluator
    logger = initialize_exp(params)
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)

    trainer.load_training_dico(params.dico_train)

    VALIDATION_METRIC_SUP = "precision_at_1-csls_knn_100"
    VALIDATION_METRIC_UNSUP = "mean_cosine-csls_knn_100-S2T-10000"

    # define the validation metric
    VALIDATION_METRIC = (
        VALIDATION_METRIC_UNSUP
        if params.dico_train == "identical_char"
        else VALIDATION_METRIC_SUP
    )
    logger.info("Validation metric: %s" % VALIDATION_METRIC)

    """
    Learning loop for Procrustes Iterative Learning
    """

    n_iter = 0
    logger.info("Starting iteration %i..." % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, "dico"):
        trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({"n_iter": n_iter})
    evaluator.all_eval(to_log)
    result_metrics = {
        "P@1-CSLS": to_log["precision_at_1-csls_knn_100"],
        "P@5-CSLS": to_log["precision_at_5-csls_knn_100"],
        "P@10-CSLS": to_log["precision_at_10-csls_knn_100"],
        "P@30-CSLS": to_log["precision_at_30-csls_knn_100"],
        "P@50-CSLS": to_log["precision_at_50-csls_knn_100"],
        "P@100-CSLS": to_log["precision_at_100-csls_knn_100"],
        "P@1-NN": to_log["precision_at_1-nn"],
        "P@5-NN": to_log["precision_at_5-nn"],
        "P@10-NN": to_log["precision_at_10-nn"],
        "P@30-NN": to_log["precision_at_30-nn"],
        "P@50-NN": to_log["precision_at_50-nn"],
        "P@100-NN": to_log["precision_at_100-nn"],
        "mean_cosin": to_log["mean_cosine-csls_knn_100-S2T-10000"],
    }

    return result_metrics


# if __name__ == "__main__":
# main
# parser = argparse.ArgumentParser(description="Supervised training")
# parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
# parser.add_argument(
#     "--verbose",
#     type=int,
#     default=0,
#     help="Verbose level (2:debug, 1:info, 0:warning)",
# )
# parser.add_argument(
#     "--exp_path",
#     type=str,
#     default="",
#     help="Where to store experiment logs and models",
# )
# parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
# parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
# parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
# parser.add_argument(
#     "--export",
#     type=str,
#     default="",
#     help="Export embeddings after training (txt / pth)",
# )

# # data
# parser.add_argument("--src_lang", type=str, default="en", help="Source language")
# parser.add_argument("--tgt_lang", type=str, default="es", help="Target language")
# parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
# parser.add_argument(
#     "--max_vocab",
#     type=int,
#     default=200000,
#     help="Maximum vocabulary size (-1 to disable)",
# )
# # training refinement
# parser.add_argument(
#     "--n_refinement",
#     type=int,
#     default=5,
#     help="Number of refinement iterations (0 to disable the refinement procedure)",
# )
# # dictionary creation parameters (for refinement)
# parser.add_argument(
#     "--dico_train",
#     type=str,
#     default="default",
#     help="Path to training dictionary (default: use identical character strings)",
# )
# parser.add_argument(
#     "--dico_eval", type=str, default="default", help="Path to evaluation dictionary"
# )
# parser.add_argument(
#     "--dico_method",
#     type=str,
#     default="csls_knn_100",
#     help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)",
# )
# parser.add_argument(
#     "--dico_build", type=str, default="S2T&T2S", help="S2T,T2S,S2T|T2S,S2T&T2S"
# )
# parser.add_argument(
#     "--dico_threshold",
#     type=float,
#     default=0,
#     help="Threshold confidence for dictionary generation",
# )
# parser.add_argument(
#     "--dico_max_rank",
#     type=int,
#     default=10000,
#     help="Maximum dictionary words rank (0 to disable)",
# )
# parser.add_argument(
#     "--dico_min_size",
#     type=int,
#     default=0,
#     help="Minimum generated dictionary size (0 to disable)",
# )
# parser.add_argument(
#     "--dico_max_size",
#     type=int,
#     default=0,
#     help="Maximum generated dictionary size (0 to disable)",
# )
# # reload pre-trained embeddings
# parser.add_argument(
#     "--src_emb", type=str, default="", help="Reload source embeddings"
# )
# parser.add_argument(
#     "--tgt_emb", type=str, default="", help="Reload target embeddings"
# )
# parser.add_argument(
#     "--normalize_embeddings",
#     type=str,
#     default="",
#     help="Normalize embeddings before training",
# )
# parser.add_argument(
#     "--load_optim", type=bool_flag, default=False, help="Reload optimal"
# )

# # parse parameters
# config = parser.parse_args()

# muse_supervised(config)
