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


def process_sentence(sentence):
    for w in sentence:
        w[1].replace("*", "").replace("+", "").replace("-", "")
    return sentence

def process_data_set():
    # same process for training & test
    data = nltk.corpus.brown.tagged_sents(categories='news')
    training_data = data[:int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)):]

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

            # Add the bigrams
            if j < len(sentence) - 1:
                next_tag = sentence[j+1][1].replace("*", "").replace("+", "").replace("-", "")
                tags[tag].increase_bigram_counter(next_tag)
            j += 1
        i += 1

    tags_set = set(tags.keys())
    tags_set.remove("START")
    return words, words["START"].corpus_size, tags, list(tags_set), test_data, training_data


def gen_transitions(tags_set, tags_dict, S):
    trans_mat = np.empty(shape=(S,S))
    start_prob, stop_prob = np.zeros(shape=(S,1)), np.zeros(shape=(S,1))
    for i in range(S):
        for j in range(S):
            trans_mat[i,j] = tags_dict[tags_set[i]].bi_prob(tags_set[j])
        start_prob[i] = tags_dict["START"].bi_prob(tags_set[i])
        stop_prob[i] = tags_dict[tags_set[i]].bi_prob("STOP")
    return start_prob, trans_mat, stop_prob


def viterbi(words_dict, corpus_size, tags_dict, tags_set, sentence : list):
    N = len(sentence)
    S = len(tags_set)
    start_prob, trans_mat, stop_prob = gen_transitions(tags_set, tags_dict, S)
    first_col = start_prob * np.array([words_dict[sentence[0]].bi_prob(i) for i in tags_set]).reshape(S,1)
    pi = np.array([-np.inf] * (S*N)).reshape(S, N)
    bp = pi.copy()
    for k in range(N):
        word = sentence[k]
        prev_col = first_col if k == 0 else pi[:,k-1]
        for j in range(S):
            tag = tags_set[j]
            trans_vec = trans_mat[:,j].reshape(S,1)
            mult_prev_col = (prev_col * trans_vec) * words_dict[word].bi_prob(tag)
            pi[j,k] = np.max(mult_prev_col)
            bp[j,k] = np.argmax(mult_prev_col)
    result_tags_idx = np.empty(shape=(N))
    last_tag = np.argmax(pi[:,-1] * stop_prob)
    result_tags_idx[-1] = last_tag
    for k in range(N-2, 0, -1):
        last_tag = bp[result_tags_idx[last_tag], k+1]
        result_tags_idx[k] = last_tag

    return [tags_set[i] for i in result_tags_idx]


if __name__ == "__main__":
    words, corpus_size, tags, tags_set, test_data, train_data = process_data_set()
    # Vietrby inference
    s1 = process_sentence(train_data[0])
    s1 = [w[0] for w in s1]
    print(s1)
    print(viterbi(words, corpus_size, tags, tags_set, sentence=s1))

