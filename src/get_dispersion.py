import os
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine

import math
import multiprocessing as mp
import torch

def vm_get_sorted_dispersion():
    """
    Calculate the dispersion of all embeddings of words/images and sorted them.
    save the result to a file.
    
    Args:
      args: includes the directories of embeddings and saving path.
    """


    categories = os.listdir(root_path)
    num_categories = len(categories)
    num_cpus = 16
    block_size = math.ceil(num_categories / num_cpus)
    p = mp.Pool(processes=num_cpus)
    
    res = [
        p.apply_async(
        func=vm_get_dispersion_multi, 
        args=(root_path, categories[i*block_size:i*block_size+block_size])) for i in range(num_cpus)
        ]

    cos_res_list = [i.get() for i in res]
    categories_dis_dict = {}
    for i in cos_res_list:
        categories_dis_dict.update(i)
    categories_dis_sorted = sorted(categories_dis_dict.items(), key = lambda kv:(kv[1], kv[0]))
    # with open('./data/sorted_dispersion_bert.txt','w') as ssw:
    with open(save_file,'w') as ssw:
        for i in categories_dis_sorted:
            ssw.write(f"{i[0]}: {i[1]}\n")
        ssw.close()


def vm_get_dispersion(root_path, embeddigns_path):
    """
    Calculating the dispersion of embeddings.
    
    Args:
      root_path: the path to the folder containing the images
      embeddigns_path: the path to the embeddings file
    """
    file_path = os.path.join(root_path, embeddigns_path)
    data = torch.load(file_path)

    obj_embeddings = data['vectors']
    
    name = data["dico"][0]

    # vectors = []
    # for _, line in enumerate(obj_embeddings[1:]):
    #     _, vect = line.rstrip().split(' ', 1)
    #     vect = np.fromstring(vect, sep=' ')
    #     # print(vect.shape)
    #     vectors.append(vect)

    relations = list(combinations(obj_embeddings, 2))
    cos_results = []
    for src, tgt in relations:
        cos_dis1 = cosine(src, tgt)
        cos_results.append(cos_dis1)

    cos_avg = np.mean(cos_results)
    return cos_avg, name

def vm_get_dispersion_multi(root_path, obj_categories):
    """
    This function takes in a root path and a list of object categories 
    to use multi-processing with some CPUs speeding up.
    
    Args:
      root_path: the path to the root directory of the dataset
      obj_categories: a list of strings, each string is the name of a category of objects
    """
    categories_dis = {}
    for obj_category in obj_categories:
        dis, name = vm_get_dispersion(root_path, obj_category)
        categories_dis[name] = dis
    return categories_dis

def lm_get_sorted_dispersion():
    data = torch.load(new_file)
    # all_words = np.array(data["dico"])
    # # unique_words = list(dict.fromkeys(data["dico"]))
    # unique_words = set(data["dico"])
    # indices = [np.where(all_words == w)[0] for w in unique_words]
    # Assuming data["dico"] is a NumPy array
    all_words = data["dico"]

    # Find unique words and their indices
    unique_words, unique_indices = np.unique(all_words, return_inverse=True)

    # Create a list of indices for each unique word
    indices = [np.where(unique_indices == i)[0] for i in range(len(unique_words))]
    
    num_cpus = 12
    block_size = math.ceil(len(unique_words) / num_cpus)
    p = mp.Pool(processes=num_cpus)
    
    res = [
        p.apply_async(
        func=lm_get_dispersion_multi, 
        args=(data, indices[i*block_size:i*block_size+block_size])) for i in range(num_cpus)
        ]
    
    cos_res_list = [i.get() for i in res]
    categories_dis_dict = {}
    for i in cos_res_list:
        categories_dis_dict.update(i)
    categories_dis_sorted = sorted(categories_dis_dict.items(), key = lambda kv:(kv[1], kv[0]))
    # with open('./data/sorted_dispersion_bert.txt','w') as ssw:
    with open(new_save_file, 'w') as ssw:
        for i in categories_dis_sorted:
            ssw.write(f"{i[0]}: {i[1]}\n")
        ssw.close()

def lm_get_dispersion(data, ids):
    """
    Calculating the dispersion of embeddings.
    
    Args:
      root_path: the path to the folder containing the images
      embeddigns_path: the path to the embeddings file
    """

    obj_embeddings = data["vectors"][ids]
    name = data["dico"][ids[0]]

    relations = list(combinations(obj_embeddings, 2))
    cos_results = []
    for src, tgt in relations:
        cos_dis1 = cosine(src, tgt)
        cos_results.append(cos_dis1)

    cos_avg = np.mean(cos_results)
    return cos_avg, name

def lm_get_dispersion_multi(all_data, categories):
    """
    This function takes in a root path and a list of object categories 
    to use multi-processing with some CPUs speeding up.
    
    Args:
      root_path: the path to the root directory of the dataset
      obj_categories: a list of strings, each string is the name of a category of objects
    """
    categories_dis = {}
    for objs in categories:
        dis, name = lm_get_dispersion(all_data, objs)
        categories_dis[name] = dis
    return categories_dis


if __name__ == "__main__":
    root_path = "/projects/nlp/people/kfb818/Dir/datasets/dispersions/vit-mae-huge"
    save_file = "./sorted_dispersion_mae.txt"

    new_file = "/projects/nlp/people/kfb818/Dir/datasets/dispersions/Llama-2-13b-hf_5120_per_sentence.pth"
    new_save_file = "./sorted_dispersion_Llama-2-13b-hf.txt"
    # vm_get_sorted_dispersion()
    lm_get_sorted_dispersion()
    # get_dispersion_multi()
    # get_dispersion()