import json
from pathlib import Path

from omegaconf import DictConfig

from src.config import ModelType
from src.utils.crawl_sentences import sentences_download
from src.utils.llm_rep_utils import LMEmbedding
from src.utils.vm_rep_utils import VMEmbedding


class RepExtractor:
    def __init__(self, config: DictConfig) -> None:
        self.config: DictConfig = config
        self.dataset_name: str = config.dataset.dataset_name
        self.image_id_pairs: str = config.dataset.image_id_pairs
        self.model_name: str = config.model.model_name
        self.model_type: ModelType = config.model.model_type
        self.model_dim: int = config.model.dim
        self.alias_emb_dir: Path = (
            Path(config.common.alias_emb_dir) / self.model_type.value
        )
        self.sentences_file: Path = Path(config.common.sentences_file)
        self.seed: int = config.common.seed

    def process_embeddings(self, config: DictConfig) -> None:
        if self.model_name == "fasttext":
            # self.__process_api_embeddings(config, dim_list)
            pass
        else:
            sentences_download(args=config)
            self.__process_embeddings()

    def __process_embeddings(self) -> None:
        if self.model_type == ModelType.VM:
            self.alias_emb_dir = self.alias_emb_dir / self.dataset_name
        if not self.alias_emb_dir.exists():
            self.alias_emb_dir.mkdir(parents=True, exist_ok=True)
        with open(self.sentences_file, "r") as f:
            sentences_array = json.load(f)
            words_array = [i for i in list(sentences_array.keys())]

        with open(self.image_id_pairs, "r") as f:
            image_id_pairs = json.load(f)
            image_ids = [i for i in list(image_id_pairs.keys())]

        save_file_path = self.alias_emb_dir / f"{self.model_name}_{self.model_dim}.pth"
        if save_file_path.exists():
            print(f"File {save_file_path} already exists.")
        elif self.model_type == ModelType.LM:
            self.__get_lm_rep(sentences_array, words_array)
        else:
            self.__get_vm_rep(image_ids)

    def __get_vm_rep(self, image_labels: list) -> None:
        embeddings_extractor = VMEmbedding(self.config, image_labels)
        embeddings_extractor.get_vm_layer_representations()

    def __get_lm_rep(self, all_sentences: dict, all_labels: list) -> None:
        if self.model_name == "fasttext":
            # self.fasttext_emb()  # to do
            return
        else:
            embeddings_extractor = LMEmbedding(self.config, all_sentences, all_labels)
            embeddings_extractor.get_lm_layer_representations()
