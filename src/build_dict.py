from pathlib import Path

# model_type = "vm"
# dispersion_ranking_path = "./sorted_dispersion_vit-mae-huge.txt"
# dispersion_ranking_path = "./sorted_dispersion_resnet152.txt"
# dispersion_ranking_path = "./sorted_dispersion_segformer-b5-finetuned-ade-640-640.txt"

# LMs
model_type = "lm"
# dispersion_ranking_path = "./sorted_dispersion_bert-large-uncased.txt"
# dispersion_ranking_path = "./sorted_dispersion_gpt2-xl.txt"
# dispersion_ranking_path = "./sorted_dispersion_opt-66b.txt"
dispersion_ranking_path = "./sorted_dispersion_Llama-2-70b-hf.txt"
model_name = dispersion_ranking_path.split("_")[-1][:-4]
data = open(dispersion_ranking_path, "r").readlines()

# keys1, keys2, keys3 = [], [], []
# for i, pair in enumerate(data):
#     if i < len(data) // 3:
#         keys1.append(pair.split(":")[0][:-2])
#     elif i < 2 * len(data) // 3:
#         keys2.append(pair.split(":")[0][:-2])
#     else:
#         keys3.append(pair.split(":")[0][:-2])

keys1, keys2, keys3 = [], [], []
for i, pair in enumerate(data):
    if i < len(data) // 3:
        keys1.append(pair.split(":")[0])
    elif i < 2 * len(data) // 3:
        keys2.append(pair.split(":")[0])
    else:
        keys3.append(pair.split(":")[0])

original_path = Path("./data/dicts/original")

for dict_path in original_path.iterdir():
    if "test" not in dict_path.name:
        continue
    save_dir = dict_path.parent.parent / (model_name+"_disp")
    save_dir.mkdir(exist_ok=True, parents=True)
    pairs = open(dict_path, "r").readlines()
    ids = [pair.split(" ")[0] for pair in pairs]
    alias = [pair.strip().split(" ")[1] for pair in pairs]
    if model_type == "vm":
        pairs1 = [pairs[i] for i, j in enumerate(ids) if j in keys1]
        pairs2 = [pairs[i] for i, j in enumerate(ids) if j in keys2]
        pairs3 = [pairs[i] for i, j in enumerate(ids) if j in keys3]
    else:
        pairs1 = [pairs[i] for i, j in enumerate(alias) if j in keys1]
        pairs2 = [pairs[i] for i, j in enumerate(alias) if j in keys2]
        pairs3 = [pairs[i] for i, j in enumerate(alias) if j in keys3]

    with open(save_dir / (dict_path.name[:-4] + "_low.txt"), "w") as f:
        f.writelines(pairs1)
    with open(save_dir / (dict_path.name[:-4] + "_medium.txt"), "w") as f:
        f.writelines(pairs2)
    with open(save_dir / (dict_path.name[:-4] + "_high.txt"), "w") as f:
        f.writelines(pairs3)

    print("="*50)
    print(dict_path.name)
    print("Number of pairs in low dispersion dict:", len(pairs1))
    print("Number of pairs in medium dispersion dict:", len(pairs2))
    print("Number of pairs in high dispersion dict:", len(pairs3))
    print("="*50)