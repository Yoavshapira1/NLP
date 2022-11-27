import nltk
from Word import *
nltk.download('brown')

# TODO: Yoav: Ojbects, save & load data, Viterbi
# TODO NADAVS: prepare training set


def add_word(dict, word, type):
    try:
        dict[word].increase_unigram_counter()
    except KeyError:
        dict[word] = Word(word) if type == Word else Tag(word)



def process_data_set():

    data = nltk.corpus.brown.tagged_sents(categories='news')
    training_data = data[:int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)):]

    # for sentence in data:
    #     print(sentence)
    # print(len(data))

    words = {}
    tags = {}

    i = 0
    while i < len(training_data):
        sentence = [("START", "START")] + data[i] + [("STOP", "STOP")]
        j = 0
        while j < len(sentence) - 1:
            word, tag = sentence[j][0], sentence[j][1].replace("*", "").replace("+", "").replace("-", "")
            add_word(dict=words, word=word, type=Word)
            add_word(dict=tags, word=tag, type=Tag)

            #add tag to Word
            words[word].increase_bigram_counter(tag)

            # add prev tag to Tag
            prev_tag = sentence[j-1][1].replace("*", "").replace("+", "").replace("-", "")
            tags[prev_tag].increase_bigram_counter(tag)
            j += 1
        i += 1
    print(tags.values())


if __name__ == "__main__":
    process_data_set()