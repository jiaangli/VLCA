import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_freq_dicts(table, pairs, name, saving_path, num_cal):
    freq_rank = {}
    alias_id_dict = {pair.strip().split(" ")[1]: pair.split(" ")[0] for pair in pairs}
    for word in alias_id_dict:
        word = word.lower().replace("_", " ")
        freq_rank[word] = table[word]

    files = [
        ("_freq10000", lambda x: 23928 < x),
        ("_freq100000", lambda x: 727 <= x <= 23928),
        ("_freq_end", lambda x: x < 727),
    ]

    for filename, condition in files:
        with open(saving_path / f"{name}{filename}.txt", "w") as file_writer:
            for idx, k in enumerate(alias_id_dict):
                k = k.lower().replace("_", " ")
                if condition(freq_rank[k]):
                    file_writer.write(f"{pairs[idx]}")
            file_writer.close()
        num_cal[filename].append(
            len(open(saving_path / f"{name}{filename}.txt", "r").readlines())
        )


if __name__ == "__main__":
    frequency_source = json.load(open("./data/aliases_freq.json", "r"))

    original_path = Path("./data/dicts/imagenet/main")

    num_cal = defaultdict(list)

    for dict_path in original_path.iterdir():
        if "test" not in dict_path.name:
            continue
        if "1k" in dict_path.name:
            continue
        save_dir = dict_path.parent.parent / "freq"
        save_dir.mkdir(exist_ok=True, parents=True)
        pairs = open(dict_path, "r").readlines()
        generate_freq_dicts(
            frequency_source, pairs, dict_path.name[:-4], original_path, num_cal
        )

    # show statistics
    for freq_type in num_cal:
        print(f"{freq_type}: {np.mean(num_cal[freq_type])}")
