import json
import nltk
import numpy as np
import re
from Word import *
nltk.download('brown')
START, STOP = "START", "STOP"

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
        if w.word not in [START, STOP]:
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


def process_tag(tag):
    tag = tag.replace("*", "")
    if "-" in tag:
        tag = tag.split("-")[0]
    if "+" in tag:
        tag = tag.split("+")[0]
    return tag


def process_sentence(pre_sentence):
    pro_sentence = []
    for j in range(len(pre_sentence)):
        word = pre_sentence[j][0]
        tag = process_tag(pre_sentence[j][1])
        pro_sentence.append((word, tag))
    return pro_sentence


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
        sentence = [(START, START)] + data[i] + [(STOP, STOP)]
        j = 0
        while j < len(sentence) - 1:
            tag = process_tag(sentence[j][1])
            word = sentence[j][0]
            add_word(dict=words, word=word, type=Word)
            add_word(dict=tags, word=tag, type=Tag)

            # add tag to Word
            words[word].increase_bigram_counter(tag)

            # Add the bigrams
            if j < len(sentence) - 1:
                next_tag = process_tag(sentence[j + 1][1])
                tags[tag].increase_bigram_counter(next_tag)
            j += 1
        i += 1

    tags_set = set(tags.keys())
    tags_set.remove(START)
    return words, words[START].corpus_size, tags, list(tags_set), test_data, training_data


def calc_probabiliteis(tags_set, tags_dict, words_dict, sentence, S, N):
    transitions, emissions = np.empty(shape=(S, S)), np.empty(shape=(S, N))
    start_prob, stop_prob = np.zeros(shape=(S, 1)), np.zeros(shape=(S, 1))

    # prepare transitions matrix
    for i in range(S):
        for j in range(S):
            transitions[i, j] = tags_dict[tags_set[i]].bi_prob(tags_set[j])
        start_prob[i] = tags_dict[START].bi_prob(tags_set[i])
        stop_prob[i] = tags_dict[tags_set[i]].bi_prob(STOP)

    # prepare emissions matrix
    for i in range(N):
        word = sentence[i]
        for j in range(S):
            tag = tags_set[j]
            try:
                # for training phase, all words should success here
                emissions[j, i] = words_dict[word].bi_counter(tag) / tags_dict[tag].uni_counter()
            except KeyError:
                # during test phase, preventing the emission vector from zeros
                emissions[j, i] = 1.

    return start_prob.reshape(S, 1), transitions, stop_prob.reshape(S, 1), emissions


def viterbi(words_dict, corpus_size, tags_dict, tags_set, sentence: list):
    N = len(sentence)
    S = len(tags_set)
    start_prob, transition_mat, stop_prob, emissions = \
        calc_probabiliteis(tags_set, tags_dict, words_dict, sentence, S, N)
    pi = np.array([-np.inf] * (S * N)).reshape(S, N)
    bp = pi.copy()
    for k in range(N):
        prev_col = start_prob if k == 0 else pi[:, k - 1].reshape(S, 1)
        emission_vec = emissions[:, k].reshape(S, 1)
        for j in range(S):
            transition_vec = transition_mat[:, j].reshape(S, 1)
            mult_prev_col = (prev_col * transition_vec) * emission_vec
            pi[j, k] = np.max(mult_prev_col)
            bp[j, k] = np.argmax(mult_prev_col)

    result_tags_idx = np.zeros(shape=(N))
    last_tag = int(np.argmax(pi[:, -1].reshape(S, 1) * stop_prob))
    result_tags_idx[-1] = last_tag
    for k in range(N - 2, 0, -1):
        last_tag = int(bp[last_tag, k])
        result_tags_idx[k] = last_tag

    return [tags_set[int(i)] for i in result_tags_idx]


def Qb_ii(words, test_data):
    known_words_counter = 0
    known_words_predicted_counter = 0
    unknown_words_counter = 0
    unknown_words_predicted_counter = 0
    total_counter = 0
    for sentence in test_data:
        sentence = process_sentence(sentence)
        sent_words = [w[0] for w in sentence]
        sent_tags = [w[1] for w in sentence]
        for word, tag in zip(sent_words, sent_tags):
            try:
                if words[word].MLE() == tag:
                    known_words_predicted_counter += 1
                known_words_counter += 1
            except KeyError:
                if "NN" == tag:
                    unknown_words_predicted_counter += 1
                unknown_words_counter += 1
            finally:
                total_counter += 1

    known_err = 1 - (known_words_predicted_counter / known_words_counter)
    unknown_err = 1 - (unknown_words_predicted_counter / unknown_words_counter)
    total_err = 1 - ((known_words_predicted_counter + unknown_words_predicted_counter) / total_counter)

    print("============== QUESTION B (ii): =============")
    print("known words error rate:  ", known_err)
    print("unknown words error rate:  ", unknown_err)
    print("overall error rate:  ", total_err)
    print()
    print()

    return total_counter


def Qc_iii(test_data, words, corpus_size, tags, tags_set):
    known_words_counter = 0
    known_words_predicted_counter = 0
    unknown_words_counter = 0
    unknown_words_predicted_counter = 0
    total_counter = 0
    for sentence in test_data:
        sentence = process_sentence(sentence)
        sent_words = [w[0] for w in sentence]
        sent_tags = [w[1] for w in sentence]
        viterbi_result = viterbi(words, corpus_size, tags, tags_set, sentence=sent_words)
        for i in range(len(sent_words)):
            real_word, real_tag, viterbi_tag = sent_words[i], sent_tags[i], viterbi_result[i]
            if real_word in words.keys():
                if viterbi_tag == real_tag:
                    known_words_predicted_counter += 1
                known_words_counter += 1
            else:
                if "NN" == real_tag:
                    unknown_words_predicted_counter += 1
                unknown_words_counter += 1
            total_counter += 1

    known_err = 1 - (known_words_predicted_counter / known_words_counter)
    unknown_err = 1 - (unknown_words_predicted_counter / unknown_words_counter)
    total_err = 1 - ((known_words_predicted_counter + unknown_words_predicted_counter) / total_counter)

    print("============== QUESTION C (iii) =============")
    print("known words error rate:  ", known_err)
    print("unknown words error rate:  ", unknown_err)
    print("overall error rate:  ", total_err)
    print(known_words_counter)
    print(known_words_predicted_counter)
    print(unknown_words_counter)
    print(unknown_words_predicted_counter)
    print()
    print()


def Qd_iii(test_data, words, new_words_set, corpus_size, tags, tags_set):
    known_words_counter = 0
    known_words_predicted_counter = 0
    unknown_words_counter = 0
    unknown_words_predicted_counter = 0
    total_counter = 0
    for sentence in test_data:
        sentence = process_sentence(sentence)
        sent_words = [w[0] for w in sentence]
        sent_tags = [w[1] for w in sentence]
        viterbi_result = viterbi(words, corpus_size, tags, tags_set, sentence=sent_words)
        for i in range(len(sent_words)):
            real_word, real_tag, viterbi_tag = sent_words[i], sent_tags[i], viterbi_result[i]
            if real_word not in new_words_set:
                if viterbi_tag == real_tag:
                    known_words_predicted_counter += 1
                known_words_counter += 1
            else:
                if viterbi_tag == real_tag:
                    unknown_words_predicted_counter += 1
                unknown_words_counter += 1
            total_counter += 1

    known_err = 1 - (known_words_predicted_counter / known_words_counter)
    unknown_err = 1 - (unknown_words_predicted_counter / unknown_words_counter)
    total_err = 1 - ((known_words_predicted_counter + unknown_words_predicted_counter) / total_counter)

    print("============== QUESTION D (iii) =============")
    print("known words error rate:  ", known_err)
    print("unknown words error rate:  ", unknown_err)
    print("overall error rate:  ", total_err)
    print(known_words_counter)
    print(known_words_predicted_counter)
    print(unknown_words_counter)
    print(unknown_words_predicted_counter)
    print()
    print()

    def pseudo_func(word : str):
        if word.endswith("ing"):
            return "ING"
        elif word.isupper():
            return "APPER"
        elif word[0].isupper():
            return "NAME"
        elif word.endswith("'s"):
            return "BELONGING"
        elif word.endswith("ed"):
            return "PASSED"
        elif "$" in word:
            return "PRICE"
        elif re.match(r"\w.\w(.\w)*", word):
            return "INITIALS"
        elif re.match(r"\d{2}[.\\-]\d{2}[.\\-]\d{4}", word):
            return "DATE"
        elif re.match(r"[A-Z\d]", word):
            return "UPPER&DIGITS"
        elif re.match(r"\d*[.-]\d*", word):
            return"DIGITS-DIGITS"
        else:
            return "UNKNOWN"


if __name__ == "__main__":


    words, corpus_size, tags, tags_set, test_data, train_data = process_data_set()
    # Qb_ii(words, test_data)
    # Qc_iii(test_data, words, corpus_size, tags, tags_set)

    # d laplace add-1
    # TODO: I am not sure if we should add 1 for each word for every tag ???????????????

    new_words = set()
    words_laplace = words.copy()
    tags_laplace = tags.copy()
    for sentence in test_data:
        for word in sentence:
            if word[0] not in words_laplace.keys():
                new_words.add(word[0])
                add_word(dict=words_laplace, word=word[0], type=Word)


    for word in words_laplace.values():
        for tag in tags_set:
            word.increase_bigram_counter(tag)

    different_words_size = len(words.keys())
    for tag in tags_laplace.values():
        tag.uni_gram_counter += different_words_size

    Qd_iii(test_data, words_laplace, new_words, corpus_size,tags_laplace,tags_set)

    #e
    for word in words.keys():
        if words[word].uni_gram_counter < 5:
            new_words.add(word)
    print(new_words)



