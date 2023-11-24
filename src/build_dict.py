import json
import random

def get_dict(file_path, seeds, id_pairs_1k=None):
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
        words = []
        shuffle_keys = list(id_pairs.keys())
        random.seed(seed)
        random.shuffle(shuffle_keys)

        for id in shuffle_keys:
            for word in id_pairs[id]:
                id_pairs_content.append(f"{id} {word}")
                words.append(word)

        train = id_pairs_content[:int(len(id_pairs_content)*0.7)]
        test = id_pairs_content[int(len(id_pairs_content)*0.7):]
        with open(f"./data/dicts/original/train_{seed}_{file_suffix}.txt", 'w') as f:
            f.write("\n".join(train))
        with open(f"./data/dicts/original/test_{seed}_{file_suffix}.txt", 'w') as f:
            f.write("\n".join(test))



if __name__ == "__main__":
    file_1k_path = "./data/id_pairs_1k.json"
    file_21k_path = "./data/id_pairs_21k.json"
    seeds = [1,2,3,4,5]

    with open(file_1k_path, 'r') as f:
        id_1k_pairs = list(json.load(f).keys())

    get_dict(file_1k_path, seeds)
    get_dict(file_21k_path, seeds, id_pairs_1k=id_1k_pairs)
