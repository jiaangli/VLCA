pairs = open("./cased_LM_all_words_potter_regression_eval.txt", "r").readlines()

# train_size = int(len(pairs) * 0.75)
train_size = int(max([int(i.split(" ")[0].split("_")[1]) for i in pairs]) * 0.70)
train_pairs = []
test_pairs = []
for i in pairs:
    if int(i.split(" ")[0].split("_")[1]) < train_size:
        train_pairs.append(i)
    else:
        test_pairs.append(i)

with open("./train.txt", "w") as f:
    for i in train_pairs:
        f.write(i)
with open("./test.txt", "w") as f:
    for i in test_pairs:
        f.write(i)
