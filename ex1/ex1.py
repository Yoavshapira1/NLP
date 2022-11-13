import spacy
from datasets import load_dataset
import numpy as np
import json
from json import JSONEncoder
nlp = spacy.load("en_core_web_sm")


class Word:

    def __init__(self, word):
        self.word = word
        self.uni_gram_counter = 0
        self.bi_gram_counters = {}
        self.max_bi_gram_counter = 0
        self.str_of_max_bi_gram_counter = "N.A"
        self.increase_unigram_counter()

    def get_uni_gram_counter(self):
        # return self.uni_gram_counter / self._corpus_size
        return self.uni_gram_counter
    def bi_prob(self, word):
        try:
            return self.bi_gram_counters[word] / self.uni_gram_counter
        except KeyError:
            return 0

    def bi_prob_log(self, word : str) -> float:
        """ Return the probability of a given string to appear after self.word"""
        p = self.bi_prob(word)
        if p == 0:
            return -np.inf
        return np.log(p)

    def MLE(self) -> str:
        """ Return the most likelihood word to appear after self.word"""
        return self.str_of_max_bi_gram_counter

    def increase_unigram_counter(self) -> None:
        self.uni_gram_counter += 1

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
               (self.word, self.uni_gram_counter, self.corpus_size, self.uni_prob_log(), self.MLE(), self.bi_prob_log(self.MLE()))


class WordEncoder(JSONEncoder):
    def default(self, obj):
        if issubclass(type(obj), Word):
            return obj.__dict__
        else:
            super().default(obj)


main_dict = dict()


def increase_uni_gram(word):
    """ Increments the counter of a single word from the corpus"""
    try:
        main_dict[word].increase_unigram_counter()
    except KeyError:
        main_dict[word] = Word(word)


def increase_bi_gram(first, second):
    """ Increments the counter of the bigram (first, second)
    NOTICE that 'first' must be in the main_dict, hence:
    ***THIS FUNCTION CAN ONLY BE CALLED AFTER increase_uni_gram(first)***"""
    main_dict[first].increase_bigram_counter(second)


def process_data_set():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for text in dataset["text"]:
        j = 0
        doc = nlp(text)

        while j < len(doc) and not doc[j].is_alpha:
            j += 1

        if j < len(doc):
            start = nlp('START')[0].lemma_
            increase_uni_gram(start)
            increase_bi_gram(start, doc[j].lemma_)

        while j < len(doc):
            increase_uni_gram(doc[j].lemma_)
            k = j + 1
            while k < len(doc):
                if doc[k].is_alpha:
                    increase_bi_gram(doc[j].lemma_, doc[k].lemma_)
                    break
                k += 1
            j = k


def increase_counter(dict, val):
    try:
        dict[val] += 1
    except:
        dict[val] = 1


def sentence_to_string_list(str):
    str = "START " + str + " STOP"
    stripped = ' '.join(x for x in str.split(" ") if x.isalpha())
    return stripped.split()


def save_data(data_dict):
    # encode
    with open('data.json', 'w') as fp:
        json.dump(data_dict, fp, indent=4, cls=WordEncoder)


def load_data(file_path):
    # decode
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    for w in data.values():
        word = Word(w["word"])
        word.uni_gram_counter = w["uni_gram_counter"]
        word.bi_gram_counters = w["bi_gram_counters"]
        word.max_bi_gram_counter = w["max_bi_gram_counter"]
        word.str_of_max_bi_gram_counter = w["str_of_max_bi_gram_counter"]
        data[w["word"]] = word
    return data


if __name__ == "__main__":
    # Psudeo code
    #
    # - for every document D:
    #     - add "START" at beginning
    #     - for every consecutive words W1, W2:
    #         add_word_to_main_dict(W1)
    #         add_word_to_bi_gram_dict(W1, W2)

    process_data_set()
    # save_data(main_dict)
    # data = load_data("data.json")
    data = main_dict

    sentences = ["Brad Pitt was born in Oklahoma", "The actor was born in USA"]
    M = 0
    # Q3
    print("Q U E S T I O N    3")
    perp_pow = 0
    for s in sentences:
        doc = nlp("START " + s)
        prob = 0
        for i in range(len(doc) - 1):
            prob += data[doc[i].lemma_].bi_prob_log(doc[i + 1].lemma_)
        perp_pow += prob
        M += len(s.split())
        print("Probability of the sentence: %s, is %f" % (s, prob))
    perplexity = np.power(np.e, -(perp_pow / M))
    print("The perplexity: %f" % perplexity)

    # Q4
    print("Q U E S T I O N    4")
    corpus_size = 0
    for val in data.values():
        if val.word != "start":
            corpus_size += val.uni_gram_counter

    perp_pow = 0
    for s in sentences:
        doc = nlp("START " + s)
        prob = 0
        for i in range(len(doc) - 1):
            bi_p = data[doc[i].lemma_].bi_prob(doc[i + 1].lemma_)
            uni_p = data[doc[i + 1].lemma_].get_uni_gram_counter() / corpus_size
            prob += np.log(((1/3) * uni_p) + ((2/3)*bi_p))
        perp_pow += prob
        print("Probability of the sentence: %s, is %f" % (s, prob))
    perplexity = np.power(np.e, -(perp_pow / M))
    print("Q4: The perplexity: %f" % perplexity)

