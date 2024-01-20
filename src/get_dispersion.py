import os
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm
import math
import multiprocessing as mp
import torch
from config import MODEL_DIM


class DispersionCalculator:
    def __init__(self, model_type, root_path, save_file, num_cpus=12):
        self.num_cpus = num_cpus
        self.model_type = model_type # LM or VM
        self.dispersion_resource = root_path # the source containing the per object embeddings (LM is a pth file, VM is a folder)
        self.save_file = save_file # the path to save the sorted dispersion

    def get_sorted_dispersion(self):
        if self.model_type == "VM":
            indices = os.listdir(self.dispersion_resource)
            block_size = math.ceil(len(indices) / self.num_cpus)
        else:
            data = torch.load(self.dispersion_resource)
            all_words = data["dico"]
            # Find unique words and their indices
            unique_words, unique_indices = np.unique(all_words, return_inverse=True)
            # Create a list of indices for each unique word
            indices = [np.where(unique_indices == i)[0] for i in range(len(unique_words))]
            block_size = math.ceil(len(unique_words) / self.num_cpus)

        p = mp.Pool(processes=self.num_cpus)

        res = [
            p.apply_async(
                func=self.get_dispersion_multi,
                args=(indices[i * block_size:i * block_size + block_size], data)
            ) for i in range(self.num_cpus)
        ]

        cos_res_list = [i.get() for i in res]
        concepts_dis_dict = {}
        for i in cos_res_list:
            concepts_dis_dict.update(i)
        concepts_dis_sorted = sorted(concepts_dis_dict.items(), key=lambda kv: (kv[1], kv[0]))

        with open(self.save_file, 'w') as ssw:
            for i in concepts_dis_sorted:
                ssw.write(f"{i[0]}: {i[1]}\n")

    def get_dispersion(self, resource_index, data=None):
        if self.model_type == "VM":
            vm_data = torch.load(os.path.join(self.dispersion_resource, resource_index))
            obj_embeddings = vm_data['vectors']
            name = vm_data["dico"][0]
        else:
            obj_embeddings = data["vectors"][resource_index]
            name = data["dico"][resource_index[0]]

        relations = list(combinations(obj_embeddings, 2))
        cos_results = []
        for src, tgt in relations:
            cos_dis1 = cosine(src, tgt)
            cos_results.append(cos_dis1)

        cos_avg = np.mean(cos_results)
        return cos_avg, name

    def get_dispersion_multi(self, sub_indices, data=None):
        concept_dis = {}
        for r_index in tqdm(sub_indices):
            dis, name = self.get_dispersion(r_index, data)
            concept_dis[name] = dis
        return concept_dis


if __name__ == "__main__":

    num_cpus = 20
    model_name = "gpt2-xl"
    model_type = "LM"
    dim = MODEL_DIM[model_type][model_name]
    if model_type == "LM":
        root_path = f"/projects/nlp/people/kfb818/Dir/datasets/LM/{model_name}_{dim}_per_object.pth" 
    else:
        root_path = f"/projects/nlp/people/kfb818/Dir/datasets/dispersion/{model_name}_images_embs"

    save_file = f"./sorted_dispersion_{model_name}.txt"

    dc = DispersionCalculator(model_type, root_path, save_file, num_cpus)
    dc.get_sorted_dispersion()
    