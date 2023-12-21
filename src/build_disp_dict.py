from pathlib import Path
import numpy as np

model_type = "vm" # "lm" or "vm"
if model_type == "vm":
    dispersion_ranking_paths = ["./data_v1/sorted_dispersion_vit-mae-huge.txt",
                            "./data_v1/sorted_dispersion_resnet152.txt",
                            "./data_v1/sorted_dispersion_segformer-b5-finetuned-ade-640-640.txt"
                            ]
else:
    dispersion_ranking_paths = ["./data_v1/sorted_dispersion_bert-large-uncased.txt",
                            "./data_v1/sorted_dispersion_gpt2-xl.txt",
                            "./data_v1/sorted_dispersion_opt-30b.txt",
                            # "./data_v1/sorted_dispersion_opt-66b.txt",
                            # "./data_v1/sorted_dispersion_Llama-2-70b-hf.txt",
                            "./data_v1/sorted_dispersion_Llama-2-13b-hf.txt"
                            ]
    
# dispersion_ranking_path = "./data_v1/sorted_dispersion_resnet152.txt"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_segformer-b5-finetuned-ade-640-640.txt"

# LMs
# model_type = "lm"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_bert-large-uncased.txt"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_gpt2-xl.txt"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_opt-66b.txt"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_opt-30b.txt"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_Llama-2-70b-hf.txt"
# dispersion_ranking_path = "./data_v1/sorted_dispersion_Llama-2-13b-hf.txt"

for dispersion_ranking_path in dispersion_ranking_paths:
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
            keys1.append(pair.split(":")[0].replace(" ","_"))
        elif i < 2 * len(data) // 3:
            keys2.append(pair.split(":")[0].replace(" ","_"))
        else:
            keys3.append(pair.split(":")[0].replace(" ","_"))

    original_path = Path("./data/dicts/original")
    lower = []
    medium = []
    higher = []
    lower_1k = []
    medium_1k = []
    higher_1k = []
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

        # print("="*50)
        # print(dict_path.name)
        # print("Number of pairs in low dispersion dict:", len(pairs1))
        # print("Number of pairs in medium dispersion dict:", len(pairs2))
        # print("Number of pairs in high dispersion dict:", len(pairs3))
        # print("="*50)

        if "1k" in dict_path.name:
            lower_1k.append(len(pairs1))
            medium_1k.append(len(pairs2))
            higher_1k.append(len(pairs3))
        else:
            lower.append(len(pairs1))
            medium.append(len(pairs2))
            higher.append(len(pairs3))

    print("*"*25+" "+model_name+" "+"*"*25)
    print("="*50)
    print("Number of pairs in low dispersion dict:", np.mean(lower))
    print("Number of pairs in medium dispersion dict:", np.mean(medium))
    print("Number of pairs in high dispersion dict:", np.mean(higher))
    print("="*50)
    print("Number of pairs in low-1k dispersion dict:", np.mean(lower_1k))
    print("Number of pairs in medium-1k dispersion dict:", np.mean(medium_1k))
    print("Number of pairs in high-1k dispersion dict:", np.mean(higher_1k))