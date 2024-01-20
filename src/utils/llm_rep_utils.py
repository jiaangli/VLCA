import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm


class LMEmbedding:
    def __init__(self, args, text_sentences_array, labels):
        self.config = args
        self.text_sentences_array = text_sentences_array
        self.pretrained_model = args.model.pretrained_model
        self.model_name = args.model.model_name
        self.labels = labels
        self.dataset_name = args.data.dataset_name
        self.alias_emb_dir = Path(args.data.alias_emb_dir) / args.model.model_type
        self.emb_per_object = True if self.model_name in ["bert-large-uncased", "gpt2-xl", "opt-30b", "Llama-2-13b-hf"] else args.data.emb_per_object
        self.device = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]

    def get_lm_layer_representations(self):
        cache_path = Path.home() / ".cache/huggingface/transformers/models" / self.model_name
        configuration = AutoConfig.from_pretrained(self.pretrained_model, cache_dir=cache_path, output_hidden_states=True)
        # configuration.pad_token_id  = 0
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, config=configuration, cache_dir=cache_path, use_fast=False, use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token

        max_memory = {k: f"{torch.cuda.get_device_properties(k).total_memory//1024**3}GB" for k in self.device}
        print(max_memory)
        if self.model_name.startswith("bert"): 
            model = AutoModel.from_pretrained(self.pretrained_model, cache_dir=cache_path, output_hidden_states=True)
            model = model.to(self.device[0])
        else:
            model = AutoModel.from_pretrained(self.pretrained_model,
                                              cache_dir=cache_path,
                                              device_map="sequential",
                                              torch_dtype=torch.float16 if not self.model_name.startswith("gpt") else None,
                                              max_memory=max_memory,
                                              config=configuration)
            # model = AutoModel.from_pretrained(self.pretrained_model,
            #                                   cache_dir=cache_path,
            #                                   output_hidden_states=True,
            #                                   device_map="sequential",
            #                                   torch_dtype=torch.float16 if not self.model_name.startswith("gpt") else None,
            #                                   max_memory=max_memory)
        model = model.eval()
        # where to store layer-wise bert embeddings of particular length
        lm_dict = {}
        pattern = r'\s+([^\w\s]+)(\s*)$'
        replacement = r'\1\2'
        # get the token embeddings
        all_words_in_context = []

        for word_idx, keys in enumerate(tqdm(self.text_sentences_array, mininterval=300, maxinterval=3600)):
            # if word_idx > 66183:
            batch_sentences = []
            for sentence in self.text_sentences_array[keys]:
                sentence = re.sub(pattern, replacement, sentence).lower()
                # sentences_words = [w for w in sentences.strip().split(' ')]
                related_alias = keys.replace("_", " ")
                all_words_in_context.append(keys)
                batch_sentences.append(sentence.strip())

            lm_dict = self.add_token_embedding_for_specific_word(batch_sentences, tokenizer, model, related_alias,
                                                                 lm_dict)
            # if word_idx == 30000:
            #     break
        self.save_embeddings(all_words_in_context, lm_dict)
        # return all_words_in_context, lm_dict

    def get_word_ind_to_token_ind(self, words_in_array, related_words, tokenizer, words_mask):
        word_ind_to_token_ind = []  # dict that maps index of word in words_in_array to index of tokens in seq_tokens
        token_ids = tokenizer(words_in_array).input_ids
        tokenized_text = tokenizer.convert_ids_to_tokens(token_ids)

        if self.model_name.startswith("bert"):
            word_tokens = tokenizer.tokenize(related_words)
        else:
            if related_words == ".22":
                match = re.search(rf"{re.escape(related_words)}[ ]?\b", words_in_array)
            # Use re.escape to escape special characters in word
            else:
                match = re.search(rf"\b{re.escape(related_words)}[ ]?\b", words_in_array)
            
            start_pos, end_pos = match.span()  # Use span to directly get start and end positions

            # Simplify logic for word_tokens
            if not self.model_name.startswith("Llama"):
                word_tokens = tokenizer.tokenize(related_words) if start_pos == 0 or not words_mask[start_pos - 1].isspace() else tokenizer.tokenize(f" {related_words}")
            else:
                word_tokens = tokenizer.tokenize(related_words) if start_pos == 0 or words_mask[start_pos - 1].isspace() else tokenizer.tokenize(f"({related_words}")[1:]

            word_ind_to_token_ind = [i for i, word in enumerate(tokenized_text) if word == word_tokens[0] and tokenized_text[i:i+len(word_tokens)] == word_tokens]

            word_ind_to_token_ind = list(range(word_ind_to_token_ind[0], word_ind_to_token_ind[0] + len(word_tokens)))

        return word_ind_to_token_ind

    def predict_lm_embeddings(self, batch_setences, tokenizer, model, lm_dict):

        indexed_tokens = tokenizer(batch_setences, return_tensors="pt", padding=True)
        indexed_tokens = indexed_tokens.to(self.device[0])
        with torch.no_grad():
            outputs = model(**indexed_tokens)

        # # Use dictionary comprehension and update method to initialize lm_dict
        # if not lm_dict:
        #     lm_dict.update({layer: [] for layer in range(len(outputs.hidden_states))})
        
        # # get all hidden states
        # return outputs.hidden_states, lm_dict

        # only get last hidden state
        if not lm_dict:
            lm_dict.update({"last": []})
        return outputs.last_hidden_state, lm_dict

    # @staticmethod
    def add_word_lm_embedding(self,lm_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):

        # layer_embedding = embeddings_to_add
        # full_sequence_embedding = embeddings_to_add.cpu().detach().numpy()
        if "bert" in self.model_name:
            lm_dict["last"].append(np.mean(embeddings_to_add[0, token_inds_to_avrg, :], 0).astype(np.float16))
        else:
            lm_dict["last"].append(embeddings_to_add[0, token_inds_to_avrg[-1], :].astype(np.float16))

        return lm_dict

    def add_token_embedding_for_specific_word(self, batch, tokenizer, model, related_alias, lm_dict, is_avg=True):
        all_sequence_embeddings, lm_dict = self.predict_lm_embeddings(batch, tokenizer, model, lm_dict)
        
        for i, sentence in enumerate(batch):
            word_ind_to_token_ind = self.get_word_ind_to_token_ind(sentence, related_alias, tokenizer, list(sentence))
            lm_dict = self.add_word_lm_embedding(lm_dict, all_sequence_embeddings[i:i+1].cpu().detach().numpy(), word_ind_to_token_ind)


        return lm_dict
    
    def save_embeddings(self, all_context_words, model_layer_dict):

        wordlist = np.array(all_context_words)
        if "bert" in self.model_name:
            wordlist = np.array([w.lower() for w in wordlist])
        unique_words = list(dict.fromkeys(wordlist))

        # print(len(model_layer_dict), len(model_layer_dict[0])) # number of layers, number of words

        layer = "last"
        if self.emb_per_object: # save per object (sentence here) embeddings
            torch.save({"dico": wordlist, "vectors": torch.from_numpy(np.vstack(model_layer_dict[layer])).float()},
                        str(self.alias_emb_dir / f"{self.model_name}_{len(model_layer_dict[layer][0])}_per_object.pth"))
        word_embeddings = self.__get_mean_word_embeddings(np.vstack(model_layer_dict[layer]), wordlist, unique_words)

        torch.save({"dico": unique_words, "vectors": torch.from_numpy(word_embeddings).float()},
            str(self.alias_emb_dir / f"{self.model_name}_{word_embeddings.shape[1]}.pth")
            )
        print(f"Saved extracted features to {str(self.alias_emb_dir)}")

    def __get_mean_word_embeddings(self, vecs, all_words, uni_words):
        word_indices = [np.where(all_words == w)[0] for w in uni_words]
        word_embeddings = np.empty((len(uni_words), vecs.shape[1]), dtype=np.float16)
        for i, indices in enumerate(word_indices):
            word_embeddings[i] = np.mean(vecs[indices], axis=0)
        return word_embeddings