import json
import torch

with open("./data/all_words.json") as json_file:
    wordlist = json.load(json_file)["all_words"]

embeddings = torch.empty((len(wordlist), 2))

frequency_source = json.load(open("/projects/nlp/data/brain/freq_wordlist_nltk", "r"))
all_words_sorted = sorted(frequency_source, key=frequency_source.get, reverse=True)
words_info = {word: i for i, word in enumerate(all_words_sorted)}

for i, word in enumerate(wordlist):
    if word not in words_info:
        words_info[word] = 1
    embeddings[i] = torch.tensor([len(word), words_info[word]])

torch.save({"dico": wordlist, "vectors": embeddings},"./data/baseline.pth")