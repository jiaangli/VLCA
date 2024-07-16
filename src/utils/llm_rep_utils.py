import re
from itertools import chain
from pathlib import Path
from typing import Any, Tuple

import duckdb
import numpy as np
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LMEmbedding:
    def __init__(self, args: DictConfig):
        self.config: DictConfig = args
        self.model_id: str = args.model.model_id
        self.bs = 32
        self.model_name: str = args.model.model_name
        self.model_dim = args.model.dim
        self.dataset_name: str = args.dataset.dataset_name
        self.alias_emb_dir: Path = (
                Path(args.common.alias_emb_dir) / args.model.model_type.value
        )
        self.emb_per_object: bool = (
            True
            if self.model_name
               in ["bert-large-uncased", "gpt2-xl", "opt-30b", "Llama-2-13b-hf"]
            else args.common.emb_per_object
        )
        self.device: list = (
            [i for i in range(torch.cuda.device_count())]
            if torch.cuda.device_count() >= 1
            else ["cpu"]
        )
        self.con = duckdb.connect(f"{self.alias_emb_dir}/{self.model_name}.db")
        self.con.execute("DROP TABLE IF EXISTS data")
        self.con.commit()
        self.con.execute(f"CREATE TABLE IF NOT EXISTS data (alias VARCHAR, embedding FLOAT[{self.model_dim}])")

    def get_lm_layer_representations(self) -> None:
        cache_path = (
                Path.home() / ".cache/huggingface/transformers/models" / self.model_id
        )
        configuration = AutoConfig.from_pretrained(
            self.model_id, cache_dir=cache_path, output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            config=configuration,
            cache_dir=cache_path,
            use_fast=False,
            token=True,
        )

        max_memory = {
            k: f"{torch.cuda.get_device_properties(k).total_memory // 1024 ** 3}GB"
            for k in self.device
        }
        if self.model_name.startswith("bert"):
            model = AutoModel.from_pretrained(
                self.model_id, cache_dir=cache_path, output_hidden_states=True
            )
            model = model.to(self.device[0])
        else:
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModel.from_pretrained(
                self.model_id,
                cache_dir=cache_path,
                device_map="sequential",
                torch_dtype=torch.float16
                if not self.model_name.startswith("gpt")
                else None,
                max_memory=max_memory,
                config=configuration,
            )
        model = model.eval()
        # where to store layer-wise bert embeddings of particular length

        # Load the text dataset
        text_sentences_array = load_dataset(
            "jaagli/common-words-79k", split="whole"
        )

        pattern = r"\s+([^\w\s]+)(\s*)$"
        replacement = r"\1\2"
        # get the token embeddings

        for i in tqdm(range(0, len(text_sentences_array), self.bs)):
            batch = text_sentences_array[i: i + self.bs]
            batch_sentences = [
                re.sub(pattern, replacement, sentence)
                for sentences in batch["sentences"]
                for sentence in sentences
            ]
            num_sentences = [len(sentences) for sentences in batch["sentences"]]
            batch_related_alias = list(chain.from_iterable([
                [word_data] * num_sentences[n] for n, word_data in enumerate(batch["alias"])
            ]))
            embeddings, related_alias = self.alias_embed(
                batch_sentences, tokenizer, model, batch_related_alias
            )

            assert len(embeddings) == len(related_alias)

            # Prepare the data for batch insert
            insert_data = [(related_alias[j], e.tolist()) for j, e in enumerate(embeddings)]
            # Execute the batch insert
            self.con.executemany("INSERT INTO data VALUES (?, ?)", insert_data)

        self.save_avg_embed(text_sentences_array["alias"])
        self.con.close()

    def map_word_to_token_ind(
            self, words_in_array: str, related_words: str, tokenizer: Any,
            words_mask: list
    ) -> list:
        """

        :param words_in_array: sentence
        :param related_words: the specific word in the sentence for which we want to obtain embeddings
        :param tokenizer:
        :param words_mask: a list of characters in the sentence
        :return: a list of token indices for the specific word
        """
        word_to_token_ind = []
        token_ids = tokenizer(words_in_array).input_ids
        tokenized_text = tokenizer.convert_ids_to_tokens(token_ids)

        if related_words == ".22":
            match = re.search(rf"{re.escape(related_words)}[ ]?\b", words_in_array.lower())
        # Use re.escape to escape special characters in word
        else:
            match = re.search(
                rf"\b{re.escape(related_words)}[ ]?\b", words_in_array.lower()
            )

        if match is None:
            return []
        # Use span to directly get start and end positions
        start_pos, end_pos = match.span()
        # to be suitable for cased and uncased
        related_words = words_in_array[start_pos: end_pos].strip()

        if self.model_name.startswith("bert"):
            word_tokens = tokenizer.tokenize(related_words)
        else:
            # Simplify logic for word_tokens
            if not self.model_name.startswith("Llama"):
                word_tokens = (
                    tokenizer.tokenize(related_words)
                    if start_pos == 0 or not words_mask[start_pos - 1].isspace()
                    else tokenizer.tokenize(f" {related_words}")
                )
            else:
                word_tokens = (
                    tokenizer.tokenize(related_words)
                    if start_pos == 0 or words_mask[start_pos - 1].isspace()
                    else tokenizer.tokenize(f"({related_words}")[1:]
                )

        for i, _ in enumerate(tokenized_text):
            if tokenized_text[i: i + len(word_tokens)] == word_tokens:
                # Decode the string based on models
                decode_str = tokenizer.decode(
                    token_ids[:i] if self.model_name.startswith("gpt") else token_ids[1:i])
                # Append to word_ind_to_token_ind based on conditions
                if (start_pos == 0 and len(decode_str) == 0) or (
                        len(decode_str) == start_pos - 1 and words_mask[start_pos - 1].isspace()) or (
                        len(decode_str) == start_pos and not words_mask[start_pos - 1].isspace()):
                    word_to_token_ind = list(range(i, i + len(word_tokens)))
                    break

        return word_to_token_ind

    def extract_sentences_embeddings(
            self, batch_sentences: list, tokenizer: Any, model: Any
    ) -> np.ndarray:
        indexed_tokens = tokenizer(batch_sentences, return_tensors="pt", padding=True)
        indexed_tokens = indexed_tokens.to(self.device[0])
        with torch.no_grad():
            outputs = model(**indexed_tokens)

        # only get last hidden state
        return outputs.last_hidden_state.detach().cpu().numpy()

    # @staticmethod
    def get_tokens_embedding(
            self,
            embeddings_to_add: np.ndarray,
            tokens_indices: list,
    ) -> np.ndarray:
        if "bert" in self.model_name:
            emb = np.mean(embeddings_to_add[0, tokens_indices, :], 0)
        else:
            emb = embeddings_to_add[0, tokens_indices[-1], :]

        return emb

    def alias_embed(
            self,
            batch: list,
            tokenizer: Any,
            model: Any,
            related_alias: list,
    ) -> Tuple[list, list]:

        all_sequence_embeddings = self.extract_sentences_embeddings(
            batch, tokenizer, model
        )

        tmp_embeddings, tmp_alias = [], []
        for i, sentence in enumerate(batch):
            w2t_indices = self.map_word_to_token_ind(
                sentence, related_alias[i].replace("_", " "), tokenizer, list(sentence)
            )
            if w2t_indices:
                tmp_emb = self.get_tokens_embedding(
                    all_sequence_embeddings[i: i + 1],
                    w2t_indices,
                )
                tmp_embeddings.append(tmp_emb)
                tmp_alias.append(related_alias[i])

        return tmp_embeddings, tmp_alias

    def save_avg_embed(self, aliases: Dataset) -> None:
        avg_embeddings, final_alias = [], []
        for i in aliases:
            query = f"SELECT embedding FROM data WHERE alias = '{i}'"
            result = self.con.execute(query).fetchall()
            if result:
                result = np.concatenate(result, axis=0)
                avg_embeddings.append(np.mean(result, axis=0))
                final_alias.append(i)
        torch.save({"dico": final_alias, "vectors": torch.from_numpy(np.array(avg_embeddings))},
                   self.alias_emb_dir / f"{self.model_name}_{self.model_dim}.pth")
