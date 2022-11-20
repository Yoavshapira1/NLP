import nltk
from Word import *
nltk.download('brown')

# TODO: Yoav: Ojbects, save & load data, Viterbi
# TODO NADAVS: prepare training set

def increase_uni_gram(dict, word):
    """ Increments the counter of a single word to a given dictionary"""
    try:
        dict[word].increase_unigram_counter()
    except KeyError:
        dict[word] = Word(word)


if __name__ == "__main__":
    words = {}
    tags = {}
    sent = "hello world tree the main data a the nadav mouse the data".split()
    tag_sent = [str(i % 3) for i in range(len(sent))]

    for word, tag in zip(sent, tag_sent):
        increase_uni_gram(words, word)
        increase_uni_gram(tags, tag)
        # add tag to Word
        words[word].increase_bigram_counter(tag)

        # add prev tag to Tag
        tags[tag].increase_bigram_counter(tag+1)