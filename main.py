from pathlib import Path

from src.utils import io_utils
from src.rep_extractor import RepExtractor
from src.procrustes import MuseExp
from src.config import ModelConfig, MODEL_DIM
from src.utils.utils_helper import reduce_dim

def get_reps(args):
    config = ModelConfig(args.model_type, args.pretrained)
    rep_extractor = RepExtractor(config=config)
    rep_extractor.process_embeddings(config)
    print("-" * 25 + "Extract and Decontextualize representation completed!" + "-" * 25)
    
    reduce_dim(config, MODEL_DIM)
    
def run_muse(args):

    # utils_helper.enforce_reproducibility(seed=config.data.seed)
    # method = config.method
    # config = utils_helper.uniform_config(args=config)
    procrustes_exp = MuseExp(args)
    procrustes_exp.run(data_split=args.data_class, extend_exp=args.more_exps)
    # procrustes_exp.run(data_split="cleaned", extend_exp="image_disp")
    pass

def main(args):
    get_reps(args)
    if args.muse:
        run_muse(args)

if __name__ == '__main__':
    args = io_utils.parser()
    main(args)