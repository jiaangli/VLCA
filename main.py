import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from src.config import MODEL_CONFIGS, RunConfig
from src.procrustes import MuseExp
from src.rep_extractor import RepExtractor
from src.utils.utils_helper import reduce_dim


def get_reps(args: DictConfig) -> None:
    rep_extractor = RepExtractor(config=args)
    rep_extractor.process_embeddings(args)
    print("-" * 25 + "Extract and Decontextualize representation completed!" + "-" * 25)

    reduce_dim(args, MODEL_CONFIGS)


def run_muse(args: DictConfig) -> None:
    procrustes_exp = MuseExp(args)
    procrustes_exp.run(data_type=args.muse.data_type, exp_type=args.muse.exp_type)


cs = ConfigStore.instance()
cs.store(name="basic_config", node=RunConfig)
for model in MODEL_CONFIGS:
    cs.store(group="model", name=f"{model}", node=MODEL_CONFIGS[model])


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(f"Run config:\n{'-'*20}\n{OmegaConf.to_yaml(cfg)}{'-'*20}\n")
    get_reps(cfg)
    if cfg.run_muse:
        run_muse(cfg)


if __name__ == "__main__":
    # args = io_utils.parser()
    main()
