import nltk
from Word import *
nltk.download('brown')

# TODO: Yoav: Ojbects, save & load data, Viterbi
# TODO NADAVS: prepare training set


def add_word(dict, word):
    """ Increments the counter of a single word to a given dictionary"""
    try:
        dict[word].increase_unigram_counter()
    except KeyError:
        dict[word] = Word(word)

def add_tag(dict, tag):
    """ Increments the counter of a single word to a given dictionary"""
    try:
        dict[tag].increase_unigram_counter()
    except KeyError:
        dict[tag] = Tag(tag)

def process_data_set():

    data = nltk.corpus.brown.tagged_sents(categories='news')
    training_data = data[:0.9 * len(data)]
    test_data = data[0.9 * len(data):]

    # for sentence in data:
    #     print(sentence)
    # print(len(data))

    words = {}
    tags = {}

    i = 0
    while i < len(training_data):
        sentence = [("START", "")] + data[i] + [("STOP", "")]
        j = 0
        while j < len(sentence) - 1:
            word, tag = sentence[j][0], sentence[j][1]
            add_word(words, word)
            add_tag(tags, tag)

            #add tag to Word
            words[word].increase_bigram_counter(tag)

            # add prev tag to Tag
            tags[tag].increase_bigram_counter(sentence[j-1][1])
            j += 1
        i += 1


if __name__ == "__main__":
    process_data_set()