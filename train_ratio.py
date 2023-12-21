import wandb
import pandas as pd

from src.config import MuseConfig
from src.procrustes import muse_supervised


if __name__ == "__main__":
    # main
    project_name = f"train_ratio_TACL"
    LM = "opt-30b"
    VM = ["vit-mae-huge", "resnet152", "segformer-b5-finetuned-ade-640-640"]
    dims = [1280, 2048, 512]

    exp_flag=''
    wandb.init(project=project_name, name=f"{LM}")
    metrics_df = pd.DataFrame()
    for idx, vm in enumerate(VM):
        emb_dim = dims[idx]
        for train_ratio in [0.1, 0.5, 1, 5, 10, 70]:
            for fold in [1,2,3,4,5]:
                metrics = {"VM": vm,
                            "LM": LM,
                            "train_ratio": train_ratio,
                            "dim": emb_dim,
                            "Fold": f"fold_{fold}"}
                muse_params = MuseConfig(exp_flag, LM, vm, emb_dim, fold, "", "cleaned").hyperparams
                muse_params.dico_train = f"/home/kfb818/projects/vislm-geo/data/dicts/ratio_exps/train_{fold}_cleaned_{train_ratio}%.txt"
                muse_params.dico_eval = f"/home/kfb818/projects/vislm-geo/data/dicts/ratio_exps/test_{fold}_cleaned_{train_ratio}%.txt"
                muse_params.src_emb = f"/projects/nlp/people/kfb818/Dir/datasets/filtered_VM/{vm}_{emb_dim}.pth"
                muse_params.tgt_emb = f"/projects/nlp/people/kfb818/Dir/datasets/filtered_LM/{LM}_{emb_dim}.pth"
                muse_res = muse_supervised(muse_params)
                metrics.update(muse_res)
                # build dataframe from dictionary
                metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])])
    wandb.log({"Results": wandb.Table(dataframe=metrics_df.round(2))}, commit=False)
    wandb.finish()
    # muse_supervised(config)

