import spacy
from datasets import load_dataset

main_dic = dict()

def process_data_set():
    nlp = spacy.load("en_core_web_sm")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    i = 0
    for text in dataset["text"]:
        doc = nlp(text)
        print(doc.text)
        print()
        i += 1
        if i == 5:
            break
    return nlp

def process_word(word):
    pass

def increase_counter(dict, val):
    try:
        dict[val] += 1
    except:
        dict[val] = 1

if __name__ == "__main__":
    ############# YOAV WAS HERE