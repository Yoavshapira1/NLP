import json

import nltk
import numpy as np

from Word import *
nltk.download('brown')

# TODO: Yoav: Viterbi
# TODO NADAVS: prepare training set


def save_data(data_dict, path):
    # encode
    with open(path, 'w') as fp:
        json.dump(data_dict, fp, indent=4, cls=WordEncoder)


def _load_data(path):
    with open(path, 'r') as fp:
        dict = json.load(fp)
    for w in dict.values():
        word = Word(w["word"])
        word.uni_gram_counter = w["uni_gram_counter"]
        word.bi_gram_counters = w["bi_gram_counters"]
        word.max_bi_gram_counter = w["max_bi_gram_counter"]
        word.str_of_max_bi_gram_counter = w["str_of_max_bi_gram_counter"]
        dict[w["word"]] = word
    return dict


def load_data(words_path, tags_path):
    # decode words data
    words = _load_data(words_path)
    corpus_size = 0
    for w in words.values():
        if w.word not in ["*", "STOP"]:
            corpus_size += w.uni_gram_counter

    # decode tags data
    tags = _load_data(tags_path)
    tags_set = set(tags.keys())
    tags_set.remove("*")
    tags_set.remove("STOP")
    return words, corpus_size, tags, list(tags_set)


def add_word(dict, word, type):
    try:
        dict[word].increase_unigram_counter()
    except KeyError:
        dict[word] = Word(word) if type == Word else Tag(word)


def build_example_sentence():
    sentence = "hello world tree the main data a the nadav mouse the data"
    processed = ("* %s STOP" % sentence).split()
    real_tags = [str(i % 3) for i in range(len(processed))]
    real_tags[0] = "*"
    real_tags[-1] = "STOP"
    return processed, real_tags


def training():
    words = {}
    tags = {}

    # Just an example for the pipline
    sentence, pos_tags = build_example_sentence()

    for i in range(len(sentence)):
        word = sentence[i]
        tag = pos_tags[i]

        # Add word and tag as unigrams
        add_word(dict=words, word=word, type=Word)
        add_word(dict=tags, word=tag, type=Tag)

        # Add the bigrams
        words[word].increase_bigram_counter(tag)
        if i < len(pos_tags) - 1:
            tags[tag].increase_bigram_counter(pos_tags[i+1])

    tags_set = set(tags.keys())
    tags_set.remove("*")
    tags_set.remove("STOP")
    return words, words["*"].corpus_size, tags, list(tags_set)


def gen_transitions(tags_set, tags_dict, S):
    trans_mat = np.empty(shape=(S,S))
    start_prob, stop_prob = np.empty(shape=(S,1)), np.empty(shape=(S,1))
    for i in range(S):
        for j in range(S):
            trans_mat[i,j] = tags_dict[tags_set[i]].bi_prob(tags_set[j])
        start_prob[i] = tags_dict["*"].bi_prob(tags_set[i])
        stop_prob[i] = tags_dict[tags_set[i]].bi_prob("STOP")
    return start_prob, trans_mat, stop_prob


def viterbi(words_dict, corpus_size, tags_dict, tags_set, sentence : list):
    N = len(sentence)
    S = len(tags_set)
    start_prob, trans_mat, stop_prob = gen_transitions(tags_set, tags_dict, S)
    first_col = start_prob * np.array([words_dict[sentence[0]].bi_prob(i) for i in tags_set]).reshape(S,1)
    pi = np.c_[first_col, np.array([-np.inf] * (S*N)).reshape(S, N)]
    bp = pi.copy()
    for k in range(1, N+1):

        word = sentence[k-1]
        prev_col = pi[:,k-1]
        for j in range(S):
            tag = tags_set[j]
            mult_prev_col = (prev_col * trans_mat[:,j]) * words_dict[word].bi_prob(tag)
            pi[k,j] = np.max(mult_prev_col)
            bp[k,j] = tags_set[np.argmax(mult_prev_col)]
    exit()
    pi[:,-1] * stop_prob



if __name__ == "__main__":
    # Transition probability of p(y|y') is given by:     y'.bi_prob[y]
    # Emission probability e(x|y) is given by:            x.bi_prob[y]
    words, corpus_size, tags, tags_set = training()

    # # save and load data example
    # save_data(words, "words.json")
    # save_data(tags, "tags.json")
    # words, corpus_size, tags, tags_set = load_data("words.json", "tags.json")

    # Vietrby inference
    viterbi(words, corpus_size, tags, tags_set, sentence=build_example_sentence()[0][1:-1])

