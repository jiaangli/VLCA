from pathlib import Path

import json


def generate_polysemy_dicts(table, pairs, name):
    num_meaning = {}
    ids = [pair.split(" ")[0] for pair in pairs]
    alias = [pair.strip().split(" ")[1].lower().replace("_", " ") for pair in pairs]
    for word in alias:
        # if len(word.split('_')) > 1:
        #     num_meaning[word] = 1
        # elif word not in table:
        #     num_meaning[word] = 0
        if word not in table:
            if len(word.split('_')) > 1:
                num_meaning[word] = 1
            else:
                num_meaning[word] = 0
        else:
            num_meaning[word] = table[word][0]
    files = [("_1", lambda x: x == 1),
                ("_over_3", lambda x: x > 3),
                ("_2_or_3", lambda x: 2 <= x <= 3)]
    for filename, condition in files:
        with open(Path("./data/dicts/poly") / f"{name}{filename}.txt", 'w') as file_writer:
            for idx, k in enumerate(alias):
                if condition(num_meaning[k]):
                    file_writer.write(f"{pairs[idx]}")
            file_writer.close()

if __name__ == "__main__":

    polysemy_source = json.load(open("/projects/nlp/data/brain/cached_requests_omw", "r"))
    words_info = polysemy_source['en']
    original_path = Path("./data/dicts/original")

    for dict_path in original_path.iterdir():
        if "test" not in dict_path.name:
            continue
        save_dir = dict_path.parent.parent / "poly"
        save_dir.mkdir(exist_ok=True, parents=True)
        pairs = open(dict_path, "r").readlines()
        generate_polysemy_dicts(words_info, pairs, dict_path.name[:-4])

