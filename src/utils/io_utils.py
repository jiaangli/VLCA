import argparse


def parser():
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument(
        "--model_type", type=str, default="LM", help="Model types: LM/VM"
    )
    parser.add_argument(
        "--pretrained", type=str, default="facebook/opt-125m", help="Model to use."
    )
    parser.add_argument("--muse", type=bool, default=False)
    parser.add_argument(
        "--exp_types", type=str, default="", help="image_disp, lang_disp, freq, poly"
    )
    parser.add_argument(
        "--data_class", type=str, default="cleaned", help="cleaned, 1k_only"
    )
    args = parser.parse_args()
    return args
