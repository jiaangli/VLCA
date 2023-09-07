import argparse
import copy

import addict
import yaml


#-----------------------------
# configs
#-----------------------------
class ForceKeyErrorDict(addict.Dict):
    def __missing__(self, name):
        raise KeyError(name)


def load_yaml(path, default_path=None):

    with open(path, encoding='utf8') as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config = ForceKeyErrorDict(**config_dict)

    if default_path is not None and path != default_path:
        with open(default_path, encoding='utf8') as default_yaml_file:
            default_config_dict = yaml.load(
                default_yaml_file, Loader=yaml.FullLoader)
            main_config = ForceKeyErrorDict(**default_config_dict)

        # simpler solution
        main_config.update(config)
        config = main_config

    return config


def update_config(config, unknown):
    # update config given args
    for idx, arg in enumerate(unknown):
        if arg.startswith("--"):
            if (':') in arg:
                k1, k2 = arg.replace("--", "").split(':')
                argtype = type(config[k1][k2])
                if argtype == bool:
                    v = unknown[idx+1].lower() == 'true'
                else:
                    if config[k1][k2] is not None:
                        v = type(config[k1][k2])(unknown[idx+1])
                    else:
                        v = unknown[idx+1]
                print(f'Changing {k1}:{k2} ---- {config[k1][k2]} to {v}')
                config[k1][k2] = v
            else:
                k = arg.replace('--', '')
                v = unknown[idx+1]
                argtype = type(config[k])
                print(f'Changing {k} ---- {config[k]} to {v}')
                config[k] = v

    return config

def save_config(datadict: ForceKeyErrorDict, path):
    datadict = copy.deepcopy(datadict)
    # datadict.training.ckpt_file = None
    # datadict.training.pop('exp_dir')
    with open(path, 'w', encoding='utf8') as outfile:
        yaml.dump(datadict.to_dict(), outfile, default_flow_style=False)

def create_args_parser():
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default=None, help='Path to config file.')
    # parser.add_argument('--method', type=str, default="Procrustes", help='Procrustes or Regression')
    # parser.add_argument("--log_path", type=str, default="",
    #                     help='the path to save experiment logs')
    # parser.add_argument("--model_prefix", type=str, default="",
    #                     help='check if there is a model prefix from huggingface')
    # parser.add_argument("--model_name", type=str, default='bert-tiny',
    #                     help='the name of words models')
    # parser.add_argument("--output_dir", type=str, default='./data/encoding_output',
    #                     help='encoding output folder path')
    # parser.add_argument("--emb_dir", type=str, default="./data/embeddings",
    #                     help='embeddings folder path')
    # parser.add_argument("--wordlist_path", type=str, default="./data/wordlist.txt",
    #                     help='the whole wordlist file path')
    # parser.add_argument("--sentences_path", type=str, default="./data/sentences.txt",
    #                     help='sentences file path')
    # parser.add_argument("--download_path", type=str, default="./data/download",
    #                     help='the path to store the crawled sentences')
    # parser.add_argument("--n_components", type=int, default=256,
    #                     help='for PCA n_components')
    # parser.add_argument("--image_dir", type=str, default="./data/imagenet_21k_small",
    #                     help='images folder path')
    # parser.add_argument("--image_classes_id", type=str, default="./data/image_id_part.txt",
    #                     help='the map of image classes and its ids')
    # parser.add_argument("--ordered_words_path", type=str, default="./data/wordlist_ordered.txt",
    #                     help='Words in the sentences in order.')
    # parser.add_argument("--encodings file path", type)
    return parser


def load_config(args, unknown, base_config_path=None):
    print("=> Parse extra configs: ", unknown)

    #---------------
    # if loading from a config file
    # use base.yaml as default
    #---------------
    config = load_yaml(args.config, default_path=base_config_path)

    # use configs given by command line to further overwrite current config
    config = update_config(config, unknown)

    # use the expname and log_root_dir to get the experiement directory
    # if 'exp_dir' not in config.training:
    #     config.training.exp_dir = os.path.join(config.training.log_root_dir, config.expname)

    # add other configs in args to config
    other_dict = vars(args)
    other_dict.pop('config')
    # other_dict.pop('resume_dir')
    config.update(other_dict)

    return config