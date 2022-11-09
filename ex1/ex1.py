import spacy
from datasets import load_dataset
import numpy as np
import json
from json import JSONEncoder

# TODO: Nadav: Process doc, Process word

class Word:

    # Automatically updating the corpus size
    _corpus_size = 0

    @property
    def corpus_size(self):
        return type(self)._corpus_size

    @corpus_size.setter
    def corpus_size(self, val):
        type(self)._corpus_size += 1

    def __init__(self, word):
        self.word = word
        self.uni_gram_counter = 0
        self.bi_gram_counters = {}
        self.max_bi_gram_counter = 0
        self.str_of_max_bi_gram_counter = "N.A"
        self.increase_unigram_counter()

    def uni_prob(self) -> float:
        """ Return the uni-gram probability"""
        return np.log(self.uni_gram_counter / self._corpus_size)

    def bi_prob(self, word : str) -> float:
        """ Return the probability of a given string to appear after self.word"""
        try:
            return np.log(self.bi_gram_counters[word] / self.uni_gram_counter)
        except KeyError:
            return 0

    def MLE(self) -> str:
        """ Return the most likelihood word to appear after self.word"""
        return self.str_of_max_bi_gram_counter

    def increase_unigram_counter(self) -> None:
        self.uni_gram_counter += 1
        if self.word != "START":
            self.corpus_size += 1

    def increase_bigram_counter(self, word : str) -> None:
        """ Adds a given word to the bigram dictionary, or increase the counters if already exists.
        Also keep tracking after the most likelihood str to appear after self.word"""
        try:
            self.bi_gram_counters[word] += 1
        except KeyError:
            self.bi_gram_counters[word] = 1
        finally:
            if self.max_bi_gram_counter < self.bi_gram_counters[word]:
                self.max_bi_gram_counter = self.bi_gram_counters[word]
                self.str_of_max_bi_gram_counter = word

    def __repr__(self) -> str:
        return "\n" \
               "Word: %s\n" \
               "Number of appearances: %d\n"\
               "Total corpus size: %d\n"\
               "Unigram probabilty: %.3f\n" \
               "Most Likey Next Word: %s\n" \
               "MLE probability: %.3f\n\n" % \
               (self.word, self.uni_gram_counter, self.corpus_size, self.uni_prob(), self.MLE(), self.bi_prob(self.MLE()))


class WordEncoder(JSONEncoder):
    def default(self, obj):
        if issubclass(type(obj), Word):
            return obj.__dict__
        else:
            super().default(obj)


main_dict = dict()


def add_word_to_main_dict(word):
    """ Increments the counter of a single word from the corpus"""
    try:
        main_dict[word].increase_unigram_counter()
    except KeyError:
        main_dict[word] = Word(word)


def add_word_to_bi_gram_dict(first, second):
    """ Increments the counter of the bigram (first, second)
    NOTICE that 'first' must be in the main_dict, hence:
    ***THIS FUNCTION CAN ONLY BE CALLED AFTER add_word_to_main_dict(first)***"""
    main_dict[first].increase_bigram_counter(second)



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

def sentence_to_string_list(str):
    str = "START " + str + " STOP"
    stripped = ' '.join(x for x in str.split(" ") if x.isalpha())
    return stripped.split()

if __name__ == "__main__":
    # Psudo code *after processing*:
    #
    # - for every document D:
    #     - add "START" at beginning and "STOP" at the end
    #     - for every consecutive words W1, W2:
    #         add_word_to_main_dict(W1)
    #         add_word_to_bi_gram_dict(W1, W2)
    # save main_dict as a JSON file!! So we don't need to run this code again

    # Pipe line example
    # just a sentence
    example = "Additionally, suppose you need to customize the functionality of any other standard dictionary method. In that case, you’ll have to override the method at hand and provide a suitable implementation that fulfills your needs. This process implies a fair amount of work. It’s also error-prone and requires advanced knowledge of Python and its data model. It can also imply performance issues because you’ll be writing the class in pure Python. The main advantage of this strategy is that the parent ABC will alert you if you miss any method in your custom implementation. For these reasons, you should embrace this strategy only if you need a dictionary-like class that’s fundamentally different from the built-in dictionary. In this tutorial, you’ll focus on creating dictionary-like classes by inheriting from the built-in dict class and the UserDict class, which seem to be the quickest and most practical strategies. Additionally, suppose you need to customize the functionality of any other standard dictionary method. a ggg In that case, you’ll have to override a ggg the method at hand and provide a ggg implementation that fulfills your needs. This process implies a fair amount of work. It’s also error-prone and requires advanced knowledge of Python and pure pure pure its data model. It can also imply performance issues because you’ll be writing the class in pure Python. The main advantage of this strategy is that the parent ABC will alert you if you miss any method in your custom implementation. For these reasons, you should embrace this strategy only if you need a dictionary-like class that’s fundamentally different from the built-in dictionary."
    # keep only alphabet words and add START & STOP
    example = sentence_to_string_list(example)
    # pipe line
    for i in range(len(example)-1):
        W1, W2 = example[i], example[i+1]
        add_word_to_main_dict(W1)
        add_word_to_bi_gram_dict(W1, W2)

    # JSON encoding & decoding example
    # encode
    with open('main_dict.json', 'w') as fp:
        json.dump(main_dict, fp, indent=4, cls=WordEncoder)

    # decode
    with open('main_dict.json', 'r') as fp:
        data = json.load(fp)
    for w in data.values():
        word = Word(w["word"])
        word.uni_gram_counter = w["uni_gram_counter"]
        word.bi_gram_counters = w["bi_gram_counters"]
        word.max_bi_gram_counter = w["max_bi_gram_counter"]
        word.str_of_max_bi_gram_counter = w["str_of_max_bi_gram_counter"]
        data[w["word"]] = word

