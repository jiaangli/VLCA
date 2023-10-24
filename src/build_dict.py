import json
import random

with open("./data/id_pairs_21k.json", 'r') as f:
    id_pairs = json.load(f)

with open("./data/id_pairs_1k.json", 'r') as f:
    id_pairs_1k = json.load(f)

id_pairs_content = []
words = []
id_1k = []

for id in id_pairs_1k:
    for word in id_pairs_1k[id]:
        id_pairs_content.append(f"{id} {word}")
        words.append(word)

# for id in id_pairs:
#     if id in id_pairs_1k:
#         continue
#     for word in id_pairs[id]:
#         id_pairs_content.append(f"{id} {word}")
#         words.append(word)

for seed in [1,2,3,4,5]:
    random.Random(seed).shuffle(id_pairs_content)
    train = id_pairs_content[:int(len(id_pairs_content)*0.7)]
    test = id_pairs_content[int(len(id_pairs_content)*0.7):]
    with open(f"./data/dicts/original/train_{seed}_1k_only.txt", 'w') as f:
        f.write("\n".join(train))
    with open(f"./data/dicts/original/test_{seed}_1k_only.txt", 'w') as f:
        f.write("\n".join(test))
