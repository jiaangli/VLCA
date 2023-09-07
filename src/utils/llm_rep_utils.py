import re
import time as tm
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class LMEmbedding:
    def __init__(self, args, text_sentences_array, labels):
        self.args = args
        self.text_sentences_array = text_sentences_array
        self.pretrained_model = args.model.pretrained_model
        self.model_name = args.model.model_name
        self.model_alias = args.model.model_alias
        self.labels = labels
        # self.is_average = args.model.i/s_avg
        self.device = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]

    def get_lm_layer_representations(self):
        cache_path = Path.home() / ".cache/huggingface/transformers/models" / self.model_name
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, cache_dir=cache_path, use_fast=False, use_auth_token=True)
        max_memory = {k: '40GB' for k in self.device}
        if self.model_alias == "bert":
            model = AutoModel.from_pretrained(self.pretrained_model, cache_dir=cache_path, output_hidden_states=True)
            model = model.to(self.device[0])
        else:
            model = AutoModel.from_pretrained(self.pretrained_model,
                                              cache_dir=cache_path,
                                              output_hidden_states=True,
                                              device_map="sequential",
                                              torch_dtype=torch.float16 if self.model_alias not in ["bert", "gpt2"] else None,
                                              max_memory=max_memory)
        model.eval()
        # where to store layer-wise bert embeddings of particular length
        lm_dict = {}
        pattern = r'\s+([^\w\s]+)(\s*)$'
        replacement = r'\1\2'
        # get the token embeddings
        start_time = tm.time()
        all_words_in_context = []
        for word_idx, keys in enumerate(tqdm(self.text_sentences_array, mininterval=300, maxinterval=3600)):
            # if word_idx >7930:
            for sentence in self.text_sentences_array[keys]:
                sentence = re.sub(pattern, replacement, sentence).lower()
                # sentences_words = [w for w in sentences.strip().split(' ')]
                related_words = keys.split("_")

                all_words_in_context.append(keys.replace("_", " "))
                lm_dict = self.add_token_embedding_for_specific_word(sentence.strip(), tokenizer, model, related_words,
                                                                    lm_dict)
                # if word_idx == 8168:
                #     break
            # if word_idx == 5:
            #     break
            # if word_idx % 10000 == 0:
            #     print(f'Completed {word_idx} out of {len(self.text_sentences_array)}: {tm.time() - start_time}')
            #     start_time = tm.time()

        return all_words_in_context, lm_dict

    def get_word_ind_to_token_ind(self, words_in_array, sentence_words, tokenizer, words_mask):
        word_ind_to_token_ind = []  # dict that maps index of word in words_in_array to index of tokens in seq_tokens
        token_ids = tokenizer(words_in_array).input_ids
        tokenized_text = tokenizer.convert_ids_to_tokens(token_ids)
        mask_tokenized_text = tokenized_text.copy()

        for i, word in enumerate(sentence_words):
            # word_ind_to_token_ind[i] = []  # initialize token indices array for current word
            if self.model_alias.startswith("bert"):
                word_tokens = tokenizer.tokenize(word)
            else:
                # Use re.escape to escape special characters in word
                match = re.search(rf"\b{re.escape(word)}[ ]?\b", ''.join(words_mask))
                start_pos, end_pos = match.span()  # Use span to directly get start and end positions

                # Simplify logic for word_tokens
                if not self.model_alias.startswith("llama"):
                    word_tokens = tokenizer.tokenize(word) if start_pos == 0 or not words_mask[start_pos - 1].isspace() else tokenizer.tokenize(f" {word}")
                else:
                    word_tokens = tokenizer.tokenize(word) if start_pos == 0 or words_mask[start_pos - 1].isspace() else tokenizer.tokenize(f"({word}")[1:]

                # Use list comprehension to replace characters in words_mask
                words_mask[start_pos: end_pos] = [" "] * (end_pos - start_pos)

            for tok in word_tokens:
                # print(sentence_words)
                ind = mask_tokenized_text.index(tok)
                word_ind_to_token_ind.append(ind)
                mask_tokenized_text[ind] = "[MASK]"

        return word_ind_to_token_ind

    def predict_lm_embeddings(self, words_in_array, tokenizer, model, lm_dict):

        indexed_tokens = tokenizer(words_in_array, return_tensors="pt")
        indexed_tokens = indexed_tokens.to(self.device[0])
        with torch.no_grad():
            outputs = model(**indexed_tokens)

        # # Use dictionary comprehension and update method to initialize lm_dict
        # if not lm_dict:
        #     lm_dict.update({layer: [] for layer in range(len(outputs.hidden_states))})
        
        # # get all hidden states
        # return outputs.hidden_states, lm_dict

        # only get last hidden state
        lm_dict.update({"last": []})
        return outputs.last_hidden_state, lm_dict

    @staticmethod
    def add_word_lm_embedding(lm_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):

        # if specific_layer >= 0:  # only add embeddings for one specified layer
        layer_embedding = embeddings_to_add
        full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        lm_dict["last"].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))

        # else:
        #     for layer, layer_embedding in enumerate(embeddings_to_add):
        #         full_sequence_embedding = layer_embedding.cpu().detach().numpy()
        #         # print(full_sequence_embedding.shape)
        #         # avrg over all tokens for specified word
        #         lm_dict[layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))

        return lm_dict

    def add_token_embedding_for_specific_word(self, word_seq, tokenizer, model, sentence_words, lm_dict, is_avg=True):
        all_sequence_embeddings, lm_dict = self.predict_lm_embeddings(word_seq, tokenizer, model, lm_dict)
        word_ind_to_token_ind = self.get_word_ind_to_token_ind(word_seq, sentence_words, tokenizer, list(word_seq))

        lm_dict = self.add_word_lm_embedding(lm_dict, all_sequence_embeddings, word_ind_to_token_ind)

        return lm_dict