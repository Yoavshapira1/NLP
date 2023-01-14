import json
import os
import pickle
import time

import nltk
import scipy

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
import numpy as np
from itertools import permutations
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx


class Arc():
    """
        An Ark obj to be sent to Chu_Liu_Edmonds_algorithm.
    """
    def __init__(self, head, tail, weight):
        self.head = head
        self.tail = tail
        self.weight = weight

    def __str__(self):
        return "(" + str(self.head) + ', ' + str(self.tail) + ', ' + str(self.weight) + ')'


class MSTparser():
    """
    An MSTparser.
    """
    def __init__(self, words_dic, pos_dict, n_iterations, bonus=False):
        self.words_dict = words_dic
        self.pos_dict = pos_dict
        self.words_dict_size = len(words_dic)
        self.pos_dict_size = len(pos_dict)
        self.bonus = bonus
        self.vec_dim = self.words_dict_size ** 2 + self.pos_dict_size ** 2
        if self.bonus:
            self.max_dist = 30
            self.num_of_adj_pos = 8
            self.vec_dim += (self.num_of_adj_pos * (self.pos_dict_size+1)) + (self.max_dist * 2) + 1 + self.pos_dict_size
        self.teta_vec = np.zeros(self.vec_dim)
        self.acumulative_teta = np.zeros(self.vec_dim)
        self.n_iteration = n_iterations
        self.mode_is_train = True

    def forward(self, t):
        max_tree = min_spanning_arborescence_nx(self.get_all_possible_arcs(t), 0)
        gold_arcs = get_gold_arcs(t)
        index_dict = self.get_vec_score_of_t(t, gold_arcs, max_tree.values())
        for index, val in index_dict.items():
            self.teta_vec[index] += val
        self.acumulative_teta += self.teta_vec

    def predict(self, t):
        max_edges = [(value.head, value.tail) for value in
                     min_spanning_arborescence_nx(self.get_all_possible_arcs(t), 0).values()]
        gold_edges = [(value.head, value.tail) for value in get_gold_arcs(t)]

        accuracy = 0
        for edge in max_edges:
            if edge in gold_edges:
                accuracy += 1
        return accuracy / len(gold_edges)

    def train_model(self, train_set):
        print("============================= START training ==============================")
        it, start = 0, time.time()
        size = len(train_set)
        for i in range(self.n_iteration):
            shuff = np.random.permutation(size)
            for s in shuff:
                self.forward(train_set[s])
                it += 1
                if it % 30 == 0:
                    print("Done: %.2f%s, time: %.2f" % ((100*it / (2*size), '%', time.time() - start)))
        self.acumulative_teta /= (self.n_iteration * size)
        print("============================= END training ==============================")

    def test_model(self, test_set):
        self.mode_is_train = False
        accuracy = 0
        for t in test_set:
            accuracy += self.predict(t)
        return accuracy / len(test_set)

    def get_feature_index(self, u, v, is_word):
        if is_word:
            return self.words_dict[u] * self.words_dict_size + self.words_dict[v]
        return self.words_dict_size ** 2 + self.pos_dict[u] * self.pos_dict_size + self.pos_dict[v]

    def get_bonus_feature_index(self, adjecant_pos_list, inbetween_pos_list, distance):
        base = self.words_dict_size ** 2 + self.pos_dict_size ** 2
        result_idx = []
        # add index corresponding to the adjacent PoS
        for adj_pos in adjecant_pos_list:
            if adj_pos is None:
                result_idx.append(base + self.pos_dict_size + 1)
            else:
                result_idx.append(base + self.pos_dict[adj_pos])
            base += self.pos_dict_size + 1

        for in_pos in inbetween_pos_list:
            result_idx.append(base + self.pos_dict[in_pos])
        base += self.pos_dict_size

        # add index for the distance between the words
        if distance <= self.max_dist and distance >= 0:
            result_idx.append(base + distance)
        elif distance >-self.max_dist and distance < 0:
            result_idx.append(base + self.max_dist + abs(distance))
        else:
            result_idx.append(base + (2 * self.max_dist))

        # add index for PoS of in-between words

        return result_idx

    def get_words_adj(self, u, v, t):
        size = len(t.nodes)
        if u['address'] - 1 >= 0:
            u_left_pos = t.nodes[u['address'] - 1]["tag"]
        else:
            u_left_pos = None

        if u['address'] - 2 >= 0:
            u_left_left_pos = t.nodes[u['address'] - 2]["tag"]
        else:
            u_left_left_pos = None

        if u['address'] + 1 < size:
            u_right_pos = t.nodes[u['address'] + 1]["tag"]
        else:
            u_right_pos = None

        if u['address'] + 2 < size:
            u_right_right_pos = t.nodes[u['address'] + 2]["tag"]
        else:
            u_right_right_pos = None

        if v['address'] - 1 >= 0:
            v_left_pos = t.nodes[v['address'] - 1]["tag"]
        else:
            v_left_pos = None

        if v['address'] - 2 >= 0:
            v_left_left_pos = t.nodes[v['address'] - 2]["tag"]
        else:
            v_left_left_pos = None

        if v['address'] + 1 < size:
            v_right_pos = t.nodes[v['address'] + 1]["tag"]
        else:
            v_right_pos = None

        if v['address'] + 2 < size:
            v_right_right_pos = t.nodes[v['address'] + 2]["tag"]
        else:
            v_right_right_pos = None

        return [u_left_pos, u_left_left_pos, u_right_pos, u_right_right_pos,
                v_left_pos, v_left_left_pos, v_right_pos, v_right_right_pos]

    def get_inbetween_pos(self, u, v, t):
        results = []
        first, second = min(u['address'], v['address']), max(u['address'], v['address'])
        if second - first >= 1:
            for address in range(second - first):
                results.append(t.nodes[first+address+1]['tag'])
        return results

    def get_bonus_index(self, u, v, t):
        adjecant_pos_list = self.get_words_adj(u, v, t)
        inbetween_pos_list = self.get_inbetween_pos(u, v, t)
        distance = u['address'] - v['address']
        bonus_index = self.get_bonus_feature_index(adjecant_pos_list, inbetween_pos_list, distance)
        return bonus_index

    def phi(self, arc, t):
        u, v = t.nodes[arc[0]], t.nodes[arc[1]]
        bonus_index = []
        if self.bonus:
            bonus_index = self.get_bonus_index(u, v, t)
        words_index = self.get_feature_index(u['word'], v['word'], is_word=True)
        pos_index = self.get_feature_index(u['tag'], v['tag'], is_word=False)
        if self.mode_is_train:
            sum = self.teta_vec[words_index] + self.teta_vec[pos_index]
            for idx in bonus_index:
                sum += self.teta_vec[idx]
            return sum
        else:
            sum = self.acumulative_teta[words_index] + self.acumulative_teta[pos_index]
            for idx in bonus_index:
                sum += self.acumulative_teta[idx]
            return sum

    def get_vec_score_of_t(self, t, arcs_gold,  arcs_max):
        result = {}
        bonus_index = []
        for arc in arcs_gold:
            word_index = self.get_feature_index(t.nodes[arc.head]["word"], t.nodes[arc.tail]["word"], is_word=True)
            tag_index = self.get_feature_index(t.nodes[arc.head]["tag"], t.nodes[arc.tail]["tag"], is_word=False)
            try:
                result[word_index] += 1
            except KeyError:
                result[word_index] = 1
            try:
                result[tag_index] += 1
            except KeyError:
                result[tag_index] = 1
            if self.bonus:
                u, v = t.nodes[arc.head], t.nodes[arc.tail]
                bonus_index = self.get_bonus_index(u, v, t)
                for idx in bonus_index:
                    try:
                        result[idx] += 1
                    except KeyError:
                        result[idx] = 1

        for arc in arcs_max:
            word_index = self.get_feature_index(t.nodes[arc.head]["word"], t.nodes[arc.tail]["word"], is_word=True)
            tag_index = self.get_feature_index(t.nodes[arc.head]["tag"], t.nodes[arc.tail]["tag"], is_word=False)
            try:
                result[word_index] -= 1
            except KeyError:
                result[word_index] = -1
            try:
                result[tag_index] -= 1
            except KeyError:
                result[tag_index] = -1
            if self.bonus:
                u, v = t.nodes[arc.head], t.nodes[arc.tail]
                bonus_index = self.get_bonus_index(u, v, t)
                for idx in bonus_index:
                    try:
                        result[idx] -= 1
                    except KeyError:
                        result[idx] = -1

        return result

    def get_all_possible_arcs(self, t):
        """
            get all possible edges to be sent to chu lie algorithm
        """
        weighted_arcs = []
        arcs = permutations(range(len(t.nodes)), 2)
        for arc in arcs:
            if arc[1] == 0:
                continue
            score = -self.phi(arc, t)
            weighted_arcs.append(Arc(t.nodes[arc[0]]["address"], t.nodes[arc[1]]["address"], score))
        return weighted_arcs


def get_dicts(corpus):
    word_set = set()
    pos_set = set()
    for sentence in corpus:
        for i in range(len(sentence.nodes)):
            word_set.add(sentence.nodes[i]["word"])
            pos_set.add(sentence.nodes[i]["tag"])
    words_dict = dict()
    pos_dict = dict()
    i = 0
    for word in word_set:
        words_dict[word] = i
        i += 1
    i = 0
    for pos in pos_set:
        pos_dict[pos] = i
        i += 1
    return words_dict, pos_dict


def get_gold_arcs(t):
    """
     This function gets a tree and return all edges.
     """
    arcs = []
    for i in range(len(t.nodes)):
        head = t.nodes[i]['head']
        if head is not None:
            arcs.append(Arc(head, i, 0))
    return arcs


if __name__ == "__main__":
    corpus = dependency_treebank.parsed_sents()
    del corpus[1854]
    train_set, test_set = corpus[:int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]

    word_dict, pos_dict = get_dicts(corpus)
    print("Corpus size: %d. Train: %d, Test: %d" % (len(corpus), len(train_set), len(test_set)))
    model = MSTparser(word_dict, pos_dict, 2)
    model_bonus = MSTparser(word_dict, pos_dict, 2, bonus=True)
    model_bonus.train_model(train_set)
    model.acumulative_teta = model_bonus.acumulative_teta[:model.vec_dim]
    print("Regular model accuracy: ", model.test_model(test_set))
    print("Bonus model accuracy: ", model_bonus.test_model(test_set))