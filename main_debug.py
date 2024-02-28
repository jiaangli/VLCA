from src.config import ModelConfig
from src.procrustes import MuseExp
from src.regression import RidgeRegression
from src.utils import io_utils


def get_reps(args):
    config = ModelConfig(args.model_type, args.pretrained)
    rr = RidgeRegression(config=config)
    rr.run_regression()
    # rep_extractor = RepExtractor(config=config)
    # rep_extractor.process_embeddings(config)
    # print("-" * 25 + "Extract and Decontextualize representation completed!" + "-" * 25)

    # reduce_dim(config, MODEL_DIM)


def run_muse(args):
    # utils_helper.enforce_reproducibility(seed=config.data.seed)
    # method = config.method
    # config = utils_helper.uniform_config(args=config)
    procrustes_exp = MuseExp(args)
    procrustes_exp.run(data_split=args.data_class, extend_exp=args.exp_types)
    # procrustes_exp.run(data_split="cleaned", extend_exp="image_disp")
    pass


def main(args):
    get_reps(args)
    if args.muse:
        run_muse(args)


if __name__ == "__main__":
    args = io_utils.parser()
    main(args)
