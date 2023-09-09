from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import threading

import fasttext as ft
import requests
import pandas as pd
import math

ft.FastText.eprint = lambda x: None

class CrawlSentences:
    def __init__(self, processID, wordlist, batch_size, download_path):
        self.processID = processID
        self.bs = batch_size
        self.download_path = download_path
        self.wordlist = wordlist

    @staticmethod
    def is_english(text, model):
        return model.predict(text)[0][0] == '__label__en'

    @staticmethod
    def check_symbols(s):
        arr = []
        SYMBOLS = {'}': '{', ']': '[', ')': '(', '>': '<'}
        SYMBOLS_L = SYMBOLS.values()
        for c in s:
            if c in SYMBOLS_L:
                # push symbol left to list
                arr.append(c)
            # pop out symbol,
            elif arr and c in SYMBOLS.keys() and arr[-1] == SYMBOLS[c]:
                arr.pop()
            else:
                pass
        if arr:
            return False
        else:
            return True


    def crawl_sentences(self, word, model):
        """
        Download sentences related to the word from wiki

        Args:
        word: the word you want to crawl
        model: the model you want to use to crawl the sentences.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'}

        sentence_list = []
        # ft_model = ft.load_model("./pretrained/lid.176.bin")

        try:
            r = requests.get(f'https://api.rhymezone.com/words?max=501&nonorm=1&k=rz_wke&rel_wke={word}',
                            headers=headers).text

            if len(r) > 2:
                try:
                    start = r.index('[{')
                    end = r.index('}]') + 2
                    data = json.loads(r[start:end])
                except Exception:
                    print(f"loaded error: {word}")
                for item in data:
                    item_info = item['word'].replace('<b>', '').replace('</b>', '').split(':', 3)
                    # print(item_info)
                    # Only select the wiki source
                    item_sentence = item_info[-1]
                    # if (item_info[0] == 'd' or item_info[0] == 'b' or item_info[0] == 'q')
                    if (item_info[0] == 'd') \
                            and len(item_sentence.split(' ')) < 16 and '...' not in item_sentence \
                            and self.check_symbols(item_sentence) and item_sentence.count('"') % 2 == 0 \
                            and item_sentence.isascii() and self.is_english(item_sentence, model):
                        # source = f'{word}: {item_sentence}'
                        sentence_list.append(item_sentence)

        except Exception:
            print(f"request error:{word}")
        return sentence_list


    def format_sentences(self, wordslist):
        """
        Create the file saving part of sentences lists. Using it to active the multi-processing to speed up.

        Args:
        wordslist: a list of words
        index: batch size of the thread
        part: the part of the corpus you want to download (1-5)
        download_path: the path to the directory where the downloaded files are stored
        """
        # with open(wordslist, 'r') as words_read:
        words = wordslist[self.processID * self.bs : (self.processID+1) * self.bs]
        examples = {}
        ft_model = ft.load_model("./ft_pretrained/lid.176.bin")
        for word_n in words:
            word = word_n.strip('\n').replace(' ', '_')
            crawled_sentences = self.crawl_sentences(word=word, model=ft_model)
            if len(crawled_sentences) >= 5:
                examples[word] = crawled_sentences[:15]
            else:
                print(word)
        return examples
    
def sentences_download(args):
    sentences_path = Path(args.data.sentences_path)
    if sentences_path.exists():
        print('Sentences already downloaded.')
    else:
        with open(Path(args.data.wordlist_path)) as json_file:
            wordlist = json.load(json_file)["all_words"]
        # wordlist = pd.read_json(str(Path(args.data.wordlist_path)))["non_digit_labels"].tolist()
        download_path = Path("./data/downloads")

        print('Begin downloading sentences.')
        if not download_path.exists():
            download_path.mkdir(exist_ok=True, parents=True)

        process_num = 32
        batch_size = math.ceil(len(wordlist) / process_num)

        with ProcessPoolExecutor(max_workers=process_num) as executor:
            processes = []
            for i in range(process_num):
                process_index = CrawlSentences(i, wordlist=wordlist, batch_size=batch_size, download_path=download_path)
                processes.append(process_index)

            # Merge dictionaries from process_results list
            process_results = list(executor.map(run_process, processes))

        # Merge dictionaries from process_results list
        merged_results = {}
        for result in process_results:
            merged_results.update(result)

        # Save merged_results to a JSON file
        output_file = Path("./data/sentences.json")
        with output_file.open("w") as f:
            json.dump(merged_results, f, indent=4)

        print('Download sentences successfully.')

def run_process(process):
    print(f"Begin: Process-{process.processID}")
    examples = process.format_sentences(process.wordlist)
    print(f"End: Process-{process.processID}")
    return examples
