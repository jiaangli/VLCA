from pathlib import Path

from src.utils import crawl_sentences, utils_helper, io_utils
from src.rep_extractor import RepExtractor
from src.procrustes import MuseExp

def get_reps():
    parser = io_utils.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = io_utils.load_config(args, unknown)
    utils_helper.enforce_reproducibility(seed=config.setup.seed)
    exp_dir = Path(f"./{config.setup.expname}")
    # method = config.method
    # config = utils_helper.uniform_config(args=config)
    crawl_sentences.sentences_download(args=config)
    rep_extractor = RepExtractor(config=config)
    rep_extractor.process_embeddings(config)
    print("-" * 25 + "Extract and Decontextualize representation completed!" + "-" * 25)
    
def run_muse():
    # procrustes_exp = MuseExp(config, train_eval="train")
    # procrustes_exp.run()
    pass

if __name__ == '__main__':
    get_reps()