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
    idx = int(0.9 * len(data))
    training_data = data[:idx]
    test_data = data[idx:]

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


def calc_probabiliteis(tags_set, tags_dict, words_dict, sentence, S, N):
    transitions, emissions = np.empty(shape=(S,S)), np.empty(shape=(S,N))
    start_prob, stop_prob = np.zeros(shape=(S,1)), np.zeros(shape=(S,1))

    # prepare transitions matrix
    for i in range(S):
        for j in range(S):
            transitions[i,j] = tags_dict[tags_set[i]].bi_prob(tags_set[j])
        start_prob[i] = tags_dict["START"].bi_prob(tags_set[i])
        stop_prob[i] = tags_dict[tags_set[i]].bi_prob("STOP")

    # prepare emissions matrix
    for i in range(N):
        word = sentence[i]
        for j in range(S):
            tag = tags_set[j]
            emissions[j,i] = words_dict[word].bi_counter(tag) / tags_dict[tag].uni_counter()

    return start_prob.reshape(S,1), transitions, stop_prob.reshape(S,1), emissions


def viterbi(words_dict, corpus_size, tags_dict, tags_set, sentence : list):
    N = len(sentence)
    S = len(tags_set)
    start_prob, transition_mat, stop_prob, emissions =\
        calc_probabiliteis(tags_set, tags_dict, words_dict, sentence, S, N)
    pi = np.array([-np.inf] * (S*N)).reshape(S, N)
    bp = pi.copy()
    for k in range(N):
        prev_col = start_prob if k == 0 else pi[:,k-1].reshape(S,1)
        emission_vec = emissions[:,k].reshape(S,1)
        for j in range(S):
            transition_vec = transition_mat[:,j].reshape(S,1)
            mult_prev_col = (prev_col * transition_vec) * emission_vec
            pi[j,k] = np.max(mult_prev_col)
            bp[j,k] = np.argmax(mult_prev_col)
    result_tags_idx = np.zeros(shape=(N))
    last_tag = int(np.argmax(pi[:,-1].reshape(S,1) * stop_prob))
    result_tags_idx[-1] = last_tag
    for k in range(N-2, 0, -1):
        last_tag = int(bp[last_tag, k+1])
        result_tags_idx[k] = last_tag

    return [tags_set[int(i)] for i in result_tags_idx]


if __name__ == "__main__":
    words, corpus_size, tags, tags_set, test_data, train_data = process_data_set()
    # # Vietrby inference
    # sentence = process_sentence(train_data[0])
    # sent_words = [w[0] for w in sentence]
    # sent_tags = [w[1] for w in sentence]
    # print(sent_tags)
    # print(viterbi(words, corpus_size, tags, tags_set, sentence=sent_words))

    # b
    known_words_counter = 0
    known_words_predicted_counter = 0
    unknown_words_counter = 0
    unknown_words_predicted_counter = 0
    for sentence in test_data:
        sentence = process_sentence(sentence)
        sent_words = [w[0] for w in sentence]
        sent_tags = [w[1] for w in sentence]
        for i in range(len(sent_words)):
            prediction = 0
            try:
                prediction = words[sent_words[i]].MLE()
                known_words_counter += 1
                if prediction == sent_tags[i]:
                    known_words_predicted_counter += 1

            except KeyError:
                prediction = "NN"
                unknown_words_counter += 1
                if prediction == sent_tags[i]:
                    unknown_words_predicted_counter += 1

    print("known words error rate:  " + str(1 - known_words_predicted_counter/known_words_counter))
    print("unknown words error rate:  " + str(1 - unknown_words_predicted_counter / unknown_words_counter))
    print("overall error rate:  " + str(1 - ((known_words_predicted_counter + unknown_words_predicted_counter)/
                                        (known_words_counter + unknown_words_counter))))

    # c
    viterbi_predicted_result = 0
    for sentence in test_data:
        sentence = process_sentence(sentence)
        sent_words = [w[0] for w in sentence]
        sent_tags = [w[1] for w in sentence]
        viterbi_result = viterbi(words, corpus_size, tags, tags_set, sentence=sent_words)
        for i in range(len(sent_words)):
            if sent_tags[i] == viterbi_result[i]:
                viterbi_predicted_result += 1

    print("viterbi error rate:  " + str(1 - (viterbi_predicted_result/(known_words_predicted_counter +
                                                                  unknown_words_predicted_counter))))







