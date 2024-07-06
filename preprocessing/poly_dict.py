import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_polysemy_dicts(table, pairs, name, saving_path, num_cal):
    num_meaning = {}
    alias_id_dict = {pair.strip().split(" ")[1]: pair.split(" ")[0] for pair in pairs}
    for word in alias_id_dict:
        if word not in table:
            num_meaning[word] = 0
        else:
            num_meaning[word] = table[word][0]
    files = [
        ("_1", lambda x: x == 1),
        ("_over_3", lambda x: x > 3),
        ("_2_or_3", lambda x: 2 <= x <= 3),
    ]
    for filename, condition in files:
        with open(saving_path / f"{name}{filename}.txt", "w") as file_writer:
            for idx, k in enumerate(alias_id_dict):
                if condition(num_meaning[k]):
                    file_writer.write(f"{pairs[idx]}")
            file_writer.close()
        num_cal[filename].append(
            len(open(saving_path / f"{name}{filename}.txt", "r").readlines())
        )


if __name__ == "__main__":
    polysemy_source = json.load(
        open("/projects/nlp/data/brain/cached_requests_omw", "r")
    )
    words_info = polysemy_source["en"]
    original_path = Path("./data/dicts/imagenet/base")
    num_cal = defaultdict(list)

    for dict_path in original_path.iterdir():
        if "test" not in dict_path.name:
            continue
        if "1k" in dict_path.name:
            continue
        save_dir = dict_path.parent.parent / "poly"
        save_dir.mkdir(exist_ok=True, parents=True)
        pairs = open(dict_path, "r").readlines()
        generate_polysemy_dicts(
            words_info, pairs, dict_path.name[:-4], original_path, num_cal
        )

    # show statistics
    for freq_type in num_cal:
        print(f"{freq_type}: {np.mean(num_cal[freq_type])}")
