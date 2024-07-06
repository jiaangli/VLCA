import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

import wandb

from .config import ExperimentsType, MuseConfig, RunConfig
from .procrustes import muse_supervised

cs = ConfigStore.instance()
cs.store(name="basic_config", node=RunConfig)


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(f"Run config:\n{'-'*20}\n{OmegaConf.to_yaml(cfg)}{'-'*20}\n")

    project_name = "train_ratio_ablation"
    LM = "opt-30b"
    VM = ["vit-mae-huge", "resnet152", "segformer-b5-finetuned-ade-640-640"]
    dims = [1280, 2048, 512]

    exp_type = ExperimentsType.BASE
    wandb.init(project=project_name, name=f"{LM}_v2")
    metrics_df = pd.DataFrame()
    for idx, vm in enumerate(VM):
        emb_dim = dims[idx]
        for train_ratio in [0.1, 0.5, 1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]:
            for fold in range(1, 6):
                metrics = {
                    "VM": vm,
                    "LM": LM,
                    "train_ratio": train_ratio,
                    "dim": emb_dim,
                    "Fold": f"fold_{fold}",
                }
                muse_params = MuseConfig(
                    exp_type=exp_type, lm=LM, vm=vm, dim=emb_dim, fold=fold, bin_name=""
                )
                muse_params.dico_train = f"{cfg.common.dictionary_path}/ratio_exps/train_{fold}_cleaned_{train_ratio}%.txt"
                muse_params.dico_eval = f"{cfg.common.dictionary_path}/ratio_exps/test_{fold}_cleaned_{train_ratio}%.txt"
                muse_params.src_emb = (
                    f"{cfg.common.alias_emb_dir}/filtered_VM/{vm}_{emb_dim}.pth"
                )
                muse_params.tgt_emb = (
                    f"{cfg.common.alias_emb_dir}/filtered_LM/{LM}_{emb_dim}.pth"
                )
                muse_res = muse_supervised(muse_params)
                metrics.update(muse_res)
                # build dataframe from dictionary
                metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])])
    wandb.log({"Results": wandb.Table(dataframe=metrics_df.round(2))}, commit=False)
    wandb.finish()


if __name__ == "__main__":
    main()
