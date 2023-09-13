import argparse

def parser():
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--model_type', type=str, default="LM", help='Model types: LM/VM')
    parser.add_argument('--pretrained', type=str, default="facebook/opt-125m", help='Model to use.')
    parser.add_argument("--muse", type=bool, default=False)
    parser.add_argument("--more_exps", type=str, default="", help="image_disp, lang_disp, freq, poly")
    parser.add_argument("--data_class", type=str, default="cleaned", help="cleaned, 1k_only")
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
    args = parser.parse_args()
    return args
