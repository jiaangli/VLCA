import os
from pathlib import Path

import fasttext
import fasttext.util
import numpy as np
import torch
import json
from sklearn.decomposition import PCA

from .utils.vm_rep_utils import VMEmbedding
from .utils.llm_rep_utils import LMEmbedding


class RepExtractor:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.data.dataset_name
        self.model_name = config.model.model_name
        self.model_alias = config.model.model_alias
        self.num_classes = config.data.num_classes
        self.model_type = config.model.model_type
        # self.suffix = "averaged" if config.model.is_avg else "first"
        self.suffix = "averaged"
        self.alias_emb_dir = Path(config.data.alias_emb_dir)/ self.suffix / self.model_name
        self.sentences_path = Path(config.data.sentences_path)
        self.seed = config.setup.seed
        self.model_dim = config.model.dim


    def process_embeddings(self, config):
        if self.num_classes > self.model_dim:
            dim_list = [self.model_dim]
        else:
            dim_list = [self.num_classes, self.model_dim]
        # print(f"Dictionary size: {len(self.pairs_dict)}")
        if self.model_name == "fasttext" or self.model_alias == "openai":
            # self.__process_api_embeddings(config, dim_list)
            pass
        else:
            self.__process_embeddings(dim_list)

    def __process_embeddings(self, dim_list):
        if not self.alias_emb_dir.exists():
            self.alias_emb_dir.mkdir(parents=True, exist_ok=True)
        with open(self.sentences_path, "r") as f:
            sentences_array = json.load(f)
            # text_sentences_array = [j for i in list(sentences_array.values()) for j in i ]
            labels_array = [i for i in list(sentences_array.keys())]

        for dim_size in dim_list:
            # for layer in range(self.config.model.n_layers):
            save_file_path = self.alias_emb_dir / f"{self.dataset_name}_{self.model_name}_dim_{dim_size}_layer_last.pth"
            if save_file_path.exists():
                print(f"File {save_file_path} already exists.")
            elif self.model_type == "LM":
                self.__get_lm_rep(sentences_array, labels_array)
            else:
                self.__get_vm_rep(labels_array)

    def __get_vm_rep(self, image_labels):
        if self.model_alias in ["sf", 'mae']:
            embeddings_extractor = VMEmbedding(self.config, image_labels)
            embeddings_extractor.get_vm_layer_representations()
        pass

    def __get_lm_rep(self, all_sentences, all_labels):
        if self.model_alias == "ft":
            self.fasttext_emb() # to do
            return
        elif self.model_alias in ["bert", "gpt2", "opt", "llama2"]:
            embeddings_extractor = LMEmbedding(self.config, all_sentences, all_labels)
            all_context_words, model_layer_dict = embeddings_extractor.get_lm_layer_representations()
        else:
            model_layer_dict = {}

        wordlist = np.array(all_context_words)
        if "bert" in self.model_name:
            wordlist = np.array([w.lower() for w in wordlist])
        unique_words = list(dict.fromkeys(wordlist))

        # print(len(model_layer_dict), len(model_layer_dict[0])) # number of layers, number of words

        layer = "last"
        embeddings = np.vstack(model_layer_dict[layer])
        if self.config.data.emb_per_sentence:
            torch.save({"dico": wordlist, "vectors": torch.from_numpy(embeddings).float()},
                        str(self.alias_emb_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_{layer}_per_sentence.pth"))
        word_embeddings = self.__get_mean_word_embeddings(embeddings, wordlist, unique_words)
        # torch.save({'dico': unique_words, 'vectors': word_embeddings}, save_file_path)
        self.__save_layer_representations(unique_words, word_embeddings, layer)
        
    def __save_layer_representations(self, words, embeddings, layer):
        # print(embeddings.shape)
        torch.save({"dico": words, "vectors": torch.from_numpy(embeddings).float()},
            str(self.alias_emb_dir / f"{self.dataset_name}_{self.model_name}_dim_{embeddings.shape[1]}_layer_{layer}.pth")
            )
        if embeddings.shape[1] > self.num_classes:
            pca = PCA(n_components=self.num_classes, random_state=self.seed)
            reduced_embeddings = pca.fit_transform(embeddings)
            torch.save({"dico": words, "vectors": torch.from_numpy(reduced_embeddings).float()},
                        str(self.alias_emb_dir / f"{self.dataset_name}_{self.model_name}_dim_{self.num_classes}_layer_{layer}.pth")
                        )
        print(f"Saved extracted features to {str(self.alias_emb_dir)}")

    def __get_mean_word_embeddings(self, vecs, all_words, uni_words):
        word_indices = [np.where(all_words == w)[0] for w in uni_words]
        word_embeddings = np.empty((len(uni_words), vecs.shape[1]))
        for i, indices in enumerate(word_indices):
            word_embeddings[i] = np.mean(vecs[indices], axis=0)
        return word_embeddings