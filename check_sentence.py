
import numpy as np

data = open('cased_LM_test_potter_fold_3.txt').readlines()

word_indx = [int(i.split(" ")[0].split("_")[1]) for i in data]
word_indx.sort()
words_diff = np.array(word_indx)
for idx,i in enumerate(np.diff(words_diff)):
    if i == 1:
        print(idx)