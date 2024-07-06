import json

from nltk.corpus import wordnet


def get_all_hyponyms(synset, level=0, max_level=None, hypo_list=[]):
    if max_level is not None and level > max_level:
        return

    hyponyms = synset.hyponyms()
    if not hyponyms:
        return

    for hyponym in hyponyms:
        # print(indentation + hyponym.name())
        hypo_list.append(hyponym.name().split(".")[0])
        get_all_hyponyms(hyponym, level + 1, max_level, hypo_list)


def get_text2id(img_label_data):
    # # get the text2id dictionary from id2text dictionary
    text2id = {}
    for i, j in img_label_data.items():
        for k in j:
            if k in text2id:
                text2id[k].append(i)
            else:
                text2id[k] = [i]
    return text2id


def remove_noisy(text2id, img_21k_data, img_1k_data):
    # noisy includes the duplicated alias for various ids, and the hypo-hyper of aliases.
    duplicated = []
    for w in text2id:
        if len(text2id[w]) > 1:
            duplicated.extend(text2id[w])
    print("Number of duplicated ids: ", len(duplicated))

    hypo_hyper_ids = []
    for w in text2id:
        word_a = wordnet.synsets(w)
        if len(word_a) == 0:
            continue
        word_a = word_a[0]
        all_hypo_list = []
        get_all_hyponyms(word_a, level=0, max_level=None, hypo_list=all_hypo_list)

        for hypo in all_hypo_list:
            if hypo in text2id:
                hypo_hyper_ids.extend(text2id[w])

    print("Number of removed ids: ", len(set(hypo_hyper_ids)))

    removed_ids = set(duplicated) | set(hypo_hyper_ids)

    for id in removed_ids:
        img_21k_data.pop(id, None)
        img_1k_data.pop(id, None)

    for id in img_1k_data:
        img_21k_data.pop(id, None)

    return img_1k_data, img_21k_data


if __name__ == "__main__":
    # 21k includes 1k data
    img_21k_data = json.load(open("./data_v2/id_pairs_21k.json", "r"))
    print("Number of ids in 21k data: ", len(img_21k_data))
    print(
        "Number of alias in 21k data: ",
        len(set([y for x in img_21k_data.values() for y in x])),
    )
    print("=" * 25)

    img_1k_data = json.load(open("./data_v2/id_pairs_1k_noisy.json", "r"))

    print("Number of ids in 1k data: ", len(img_1k_data))
    print(
        "Number of alias in 1k data: ",
        len(set([y for x in img_1k_data.values() for y in x])),
    )
    print("=" * 25)

    # union of two dictionaries
    all_img_data = img_1k_data | img_21k_data

    text2id_all = get_text2id(img_21k_data)

    # clean_21k doesn't include the noisy and 1k data here
    clean_1k, clean_21k = remove_noisy(text2id_all, img_21k_data, img_1k_data)

    print("Number of ids in 21k data: ", len(clean_21k))
    print(
        "Number of alias in 21k data: ",
        len(set([y for x in clean_21k.values() for y in x])),
    )
    print("=" * 25)
    print("Number of ids in 1k data: ", len(clean_1k))
    print(
        "Number of alias in 1k data: ",
        len(set([y for x in clean_1k.values() for y in x])),
    )
