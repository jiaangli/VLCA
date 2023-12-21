import json
import random
from pathlib import Path


def write_data_to_file(data, filename):
    with open(filename, 'w') as f:
        f.write("\n".join(data))

def get_dict(file_path, seeds, id_pairs_1k=None, ratio_flag=False, save_root=Path("./data/dicts/original")):
    with open(file_path, 'r') as f:
        id_pairs = json.load(f)

    if "21k" in file_path:
        file_suffix = "cleaned"
        for id in id_pairs_1k:
            id_pairs.pop(id)
    else:
        file_suffix = "1k_only"

    for seed in seeds:
        id_pairs_content = []
        # words = []
        shuffle_keys = list(id_pairs.keys())
        random.seed(seed)
        random.shuffle(shuffle_keys)

        ratios = [0.1, 0.5, 1, 5, 10, 70] if ratio_flag else [70]
        
        for ratio in ratios:
            train, test = [], []
            for idx, id in enumerate(shuffle_keys):
                data_list = train if idx <= len(shuffle_keys) * ratio / 100 else test

                for word in id_pairs[id]:
                    data_list.append(f"{id} {word}")

            ratio_mark = f"_{ratio}%" if ratio_flag else ""

            write_data_to_file(train, save_root / f"train_{seed}_{file_suffix}{ratio_mark}.txt")
            write_data_to_file(test, save_root / f"test_{seed}_{file_suffix}{ratio_mark}.txt")



if __name__ == "__main__":
    file_1k_path = "./data/id_pairs_1k.json"
    file_21k_path = "./data/id_pairs_21k.json"
    seeds = [1,2,3,4,5]
    ratio_flag = True
    save_root = Path("./data/dicts/ratio_exps") if ratio_flag else Path("./data/dicts/original")
    save_root.mkdir(exist_ok=True, parents=True)

    with open(file_1k_path, 'r') as f:
        id_1k_pairs = list(json.load(f).keys())

    # get_dict(file_1k_path, seeds, ratio_flag=ratio_flag, save_root=save_root) # 1k dictionaries
    get_dict(file_21k_path, seeds, id_pairs_1k=id_1k_pairs, ratio_flag=ratio_flag, save_root=save_root) # 21k dictionaries
