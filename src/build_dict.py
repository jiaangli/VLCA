import json

with open("./data/.json") as json_file:
    wordlist = json.load(json_file).keys()