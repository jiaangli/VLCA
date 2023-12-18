from pathlib import Path
import json

from src.utils.vm_rep_utils import VMEmbedding
from src.utils.llm_rep_utils import LMEmbedding
from src.utils.crawl_sentences import sentences_download


class RepExtractor:
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.data.dataset_name
        self.model_name = config.model.model_name
        self.num_classes = config.data.num_classes
        self.model_type = config.model.model_type
        self.alias_emb_dir = Path(config.data.alias_emb_dir) / self.model_type
        self.sentences_path = Path(config.data.sentences_path)
        self.seed = config.seed
        self.model_dim = config.model.dim
        self.image_id_pairs = config.data.image_id_pairs


    def process_embeddings(self, config):
        if self.model_name == "fasttext":
            # self.__process_api_embeddings(config, dim_list)
            pass
        else:
            sentences_download(args=config)
            self.__process_embeddings()

    def __process_embeddings(self):
        if self.model_type == "VM":
            self.alias_emb_dir = self.alias_emb_dir / self.dataset_name
        if not self.alias_emb_dir.exists():
            self.alias_emb_dir.mkdir(parents=True, exist_ok=True)
        with open(self.sentences_path, "r") as f:
            sentences_array = json.load(f)
            # text_sentences_array = [j for i in list(sentences_array.values()) for j in i ]
            words_array = [i for i in list(sentences_array.keys())]
        
        with open(self.image_id_pairs, "r") as f:
            image_id_pairs = json.load(f)
            image_ids = [i for i in list(image_id_pairs.keys())]

        save_file_path = self.alias_emb_dir / f"{self.model_name}_{self.model_dim}.pth"
        if save_file_path.exists():
            print(f"File {save_file_path} already exists.")
        elif self.model_type == "LM":
            self.__get_lm_rep(sentences_array, words_array)
        else:
            self.__get_vm_rep(image_ids)

    def __get_vm_rep(self, image_labels):
        if not self.model_name.startswith("res"):
            embeddings_extractor = VMEmbedding(self.config, image_labels)
            embeddings_extractor.get_vm_layer_representations()
        pass

    def __get_lm_rep(self, all_sentences, all_labels):
        if self.model_name == "fasttext":
            self.fasttext_emb() # to do
            return
        else:
            embeddings_extractor = LMEmbedding(self.config, all_sentences, all_labels)
            embeddings_extractor.get_lm_layer_representations()