from json import JSONEncoder
from abc import abstractmethod
import numpy as np


class WordAbs:

    def __init__(self, word):
        self.word = word
        self.uni_gram_counter = 0
        self.bi_gram_counters = {}
        self.max_bi_gram_counter = 0
        self.str_of_max_bi_gram_counter = "N.A"
        self.increase_unigram_counter()

    @abstractmethod
    def increase_unigram_counter(self):
        raise NotImplementedError("increase_unigram_counter must ve implemented")

    def uni_prob(self, corpus_size) -> float:
        """ Return the uni-gram probability"""
        return np.log(self.uni_gram_counter / corpus_size)

    def bi_prob(self, word : str) -> float:
        """ Return the probability of a given string to appear after self.word"""
        try:
            return np.log(self.bi_gram_counters[word] / self.uni_gram_counter)
        except KeyError:
            return -np.inf

    def MLE(self) -> str:
        """ Return the most likelihood word to appear after self.word"""
        return self.str_of_max_bi_gram_counter

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
               "Most Likey Next Word: %s\n" \
               "MLE probability: %.3f\n\n" % \
               (self.word, self.uni_gram_counter, self.MLE(), self.bi_prob(self.MLE()))

class Word(WordAbs):

    # Automatically updating the corpus size
    _corpus_size = 0

    @property
    def corpus_size(self):
        return type(self)._corpus_size

    @corpus_size.setter
    def corpus_size(self, val):
        type(self)._corpus_size += 1

    def __init__(self, word):
        super().__init__(word)
        self.str_of_max_bi_gram_counter = "NN"

    def increase_unigram_counter(self) -> None:
        self.uni_gram_counter += 1
        if not self.word in ["*", "STOP"]:
            self.corpus_size += 1


class Tag(WordAbs):
    def increase_unigram_counter(self) -> None:
        self.uni_gram_counter += 1


class WordEncoder(JSONEncoder):
    def default(self, obj):
        if issubclass(type(obj), WordAbs):
            return obj.__dict__
        else:
            super().default(obj)