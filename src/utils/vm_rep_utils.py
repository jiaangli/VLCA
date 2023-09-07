import re
import time as tm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_classes, extractor, resolution=224) -> None:
        super(ImageDataset, self).__init__()
        self.image_root = image_dir
        self.labels = image_classes
        # self.names = [i.strip('\n').split(': ')[1] for i in self.id_name_pairs]
        self.extractor = extractor
        self.MAX_SIZE = 50
        self.RESOLUTION_HEIGHT = resolution
        self.RESOLUTION_WIDTH = resolution
        self.CHANNELS = 3

    def __len__(self):
        # return len(self.id_name_pairs)
        return len(self.labels)
    
    def __getitem__(self, index):
        images = []
        category_path = self.image_root / self.labels[index]
        for filename in category_path.iterdir():
            try:
                images.append(Image.open(category_path / filename).convert('RGB'))
            except:
                print('Failed to pil', filename)

        category_size = len(images)
        values = self.extractor(images=images, return_tensors="pt")
        inputs = torch.zeros(self.MAX_SIZE, self.CHANNELS, self.RESOLUTION_HEIGHT, self.RESOLUTION_WIDTH)
        with torch.no_grad():
            inputs[:category_size,:,:,:].copy_(values.pixel_values)

        return inputs, (self.labels[index], category_size)

class VMEmbedding:
    def __init__(self, args, labels):
        self.args = args
        self.labels = labels
        self.pretrained_model = args.model.pretrained_model
        self.model_alias = args.model.model_alias
        self.image_dir = Path(args.data.image_dir)
        self.model_name = args.model.model_name
        self.bs = 8
        self.n_layers = args.model.n_layers
        self.alias_emb_dir = Path(args.data.alias_emb_dir)/ "averaged" / self.model_name
        # self.is_average = args.model.i/s_avg
        self.device = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]

    def get_vm_layer_representations(self):
        cache_path = Path.home() / ".cache/huggingface/transformers/models" / self.pretrained_model
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.pretrained_model, cache_dir=cache_path)
        if self.model_alias != "resnet":
            model = AutoModel.from_pretrained(self.pretrained_model, cache_dir=cache_path, output_hidden_states=True, return_dict=True)
            model = model.to(self.device[0])

        model.eval()

        imageset = ImageDataset(self.image_dir, self.labels, feature_extractor )
        image_dataloader = torch.utils.data.DataLoader(imageset, batch_size=self.bs, num_workers=4, pin_memory=True)

        # where to store layer-wise bert embeddings of particular length
        vm_dict = {}

        # images_name = []
        categories_encode = {}
        if not categories_encode:
            categories_encode.update({layer: [] for layer in range(self.n_layers+1)})
        image_categories = []

        for inputs, (names, category_size) in tqdm(image_dataloader, mininterval=60.0, maxinterval=360.0):
            inputs_shape = inputs.shape
            inputs = inputs.reshape(-1, inputs_shape[2],inputs_shape[3],inputs_shape[4]).to(self.device[0])
            
            with torch.no_grad():
                outputs = model(pixel_values=inputs)
                # chunks = torch.chunk(outputs.last_hidden_state[:,0,:].cpu(), inputs_shape[0], dim=0)
            for layer_i in range(self.n_layers+1):
                if layer_i < self.n_layers:
                    chunks = torch.chunk(outputs.hidden_states[layer_i][:,1:,:].cpu(), inputs_shape[0], dim=0)
                else:
                    chunks = torch.chunk(outputs.last_hidden_state[:,1:,:].cpu(), inputs_shape[0], dim=0)

                for idx, chip in enumerate(chunks):
                    # features for every image
                    # images_features = np.mean(chip[:category_size[idx]].numpy(), axis=(2,3), keepdims=True).squeeze()
                    images_features = chip[:category_size[idx]].numpy()
                    # features for categories
                    category_feature = np.expand_dims(images_features.mean(axis=0), 0)
                    if layer_i == 0:
                        image_categories.append(names[idx])
                    # images_name = [f"{names[idx]}_{i}" for i in range(category_size[idx])]

                    categories_encode[layer_i].append(category_feature)

        for layer in categories_encode.keys():
            embeddings = np.vstack(categories_encode[layer])
        #   categories_encode = np.concatenate(categories_encode)
            dim_size = embeddings.shape[1]

            torch.save({"dico": image_categories, "vectors": torch.from_numpy(embeddings).float()}, 
                        str(self.alias_emb_dir / f"things_{self.model_name}_dim_{dim_size}_layer_{layer}.pth"))
            
            














    #     pattern = r'\s+([^\w\s]+)(\s*)$'
    #     replacement = r'\1\2'
    #     # get the token embeddings
    #     start_time = tm.time()
    #     all_words_in_context = []
    #     for sent_idx, sentences in enumerate(tqdm(self.text_sentences_array, mininterval=300, maxinterval=3600)):
    #         sentences = re.sub(pattern, replacement, sentences)

    #         sentences_words = [w for w in sentences.strip().split(' ')]
    #         all_words_in_context.extend(sentences_words)
    #         lm_dict = self.add_token_embedding_for_specific_word(sentences.strip(), tokenizer, model, sentences_words,
    #                                                              lm_dict)

    #         if sent_idx % 10000 == 0:
    #             print(f'Completed {sent_idx} out of {len(self.text_sentences_array)}: {tm.time() - start_time}')
    #             start_time = tm.time()

    #     return all_words_in_context, lm_dict

    # def get_word_ind_to_token_ind(self, words_in_array, sentence_words, tokenizer, words_mask):
    #     word_ind_to_token_ind = {}  # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    #     token_ids = tokenizer(words_in_array).input_ids
    #     tokenized_text = tokenizer.convert_ids_to_tokens(token_ids)
    #     mask_tokenized_text = tokenized_text.copy()

    #     for i, word in enumerate(sentence_words):
    #         word_ind_to_token_ind[i] = []  # initialize token indices array for current word
    #         if self.model_alias.startswith("bert"):
    #             word_tokens = tokenizer.tokenize(word)
    #         else:
    #             # Use re.escape to escape special characters in word
    #             match = re.search(rf"\b{re.escape(word)}[ ]?\b", ''.join(words_mask))
    #             start_pos, end_pos = match.span()  # Use span to directly get start and end positions

    #             # Simplify logic for word_tokens
    #             word_tokens = tokenizer.tokenize(word) if start_pos == 0 or words_mask[start_pos - 1] in (
    #                 '(', '\"', '-', '\'', "‘") else tokenizer.tokenize(f" {word}")

    #             # Use list comprehension to replace characters in words_mask
    #             words_mask[start_pos: end_pos] = [" "] * (end_pos - start_pos)

    #         for tok in word_tokens:
    #             ind = mask_tokenized_text.index(tok)
    #             word_ind_to_token_ind[i].append(ind)
    #             mask_tokenized_text[ind] = "[MASK]"

    #     return word_ind_to_token_ind

    # def predict_lm_embeddings(self, words_in_array, tokenizer, model, lm_dict):

    #     indexed_tokens = tokenizer(words_in_array, return_tensors="pt")
    #     indexed_tokens = indexed_tokens.to(self.device[0])
    #     with torch.no_grad():
    #         outputs = model(**indexed_tokens)

    #     # Use dictionary comprehension and update method to initialize lm_dict
    #     if not lm_dict:
    #         lm_dict.update({layer: [] for layer in range(len(outputs.hidden_states))})
    #     return outputs.hidden_states, lm_dict

    # @staticmethod
    # def add_word_lm_embedding(lm_dict, embeddings_to_add, token_inds_to_avrg, specific_layer=-1):

    #     if specific_layer >= 0:  # only add embeddings for one specified layer
    #         layer_embedding = embeddings_to_add[specific_layer]
    #         full_sequence_embedding = layer_embedding.cpu().detach().numpy()
    #         lm_dict[specific_layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))
    #     else:
    #         for layer, layer_embedding in enumerate(embeddings_to_add):
    #             full_sequence_embedding = layer_embedding.cpu().detach().numpy()
    #             # print(full_sequence_embedding.shape)
    #             # avrg over all tokens for specified word
    #             lm_dict[layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avrg, :], 0))

    #     return lm_dict

    # def add_token_embedding_for_specific_word(self, word_seq, tokenizer, model, sentence_words, lm_dict, is_avg=True):
    #     all_sequence_embeddings, lm_dict = self.predict_lm_embeddings(word_seq, tokenizer, model, lm_dict)
    #     word_ind_to_token_ind = self.get_word_ind_to_token_ind(word_seq, sentence_words, tokenizer, list(word_seq))

    #     for token_inds_to_avrg in list(word_ind_to_token_ind.keys()):
    #         if is_avg:
    #             token_ind = word_ind_to_token_ind[token_inds_to_avrg]
    #         else:
    #             # only use the first token
    #             token_ind = [word_ind_to_token_ind[token_inds_to_avrg][0]]
    #         lm_dict = self.add_word_lm_embedding(lm_dict, all_sequence_embeddings, token_ind)

    #     return lm_dict