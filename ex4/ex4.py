import nltk

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
import numpy as np
from itertools import permutations
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx


class MSTparser():
    """
    An MSTparser.
    """
    def __init__(self, words_dic, pos_dict, n_iterations):
        self.words_dict = words_dic
        self.pos_dict = pos_dict
        self.words_dict_size = len(words_dic)
        self.pos_dict_size = len(pos_dict)
        self.vec_dim = self.words_dict_size ** 2 + self.pos_dict_size ** 2
        self.teta_vec = np.zeros(self.vec_dim, dtype=float) # dtype = float ???????????????
        self.n_iteration = n_iterations

    def forward(self, t):
        max_tree = min_spanning_arborescence_nx(self.get_all_possible_edges(t))

        # TO DO:  update teta according to gold T

    def predict(self, t):
        pass

    def train_model(self, train_set):
        for i in range(self.n_iteration):
            for t in train_set:
                self.forward(t)



    def test_model(self, test_set):
        pass

    def get_feature_index(self, u, v, is_word):
        if is_word:
            return self.words_dict[u] * self.words_dict_size + self.words_dict[v]
        return self.words_dict_size ** 2 + self.pos_dict[u] * self.pos_dict_size + self.pos_dict[v]

    def phi(self, u, v):
        vec = np.zeros(self.vec_dim, dtype=bool)
        words_index = self.get_feature_index(u['word'], v['word'], True)
        pos_index = self.get_feature_index(u['tag'], v['tag'], False)
        vec[words_index] = True
        vec[pos_index] = True
        return vec

    def get_all_possible_edges(self, t):
        """
            get all possible edges to be sent to chu lie algorithm
        """
        temp_dict = t.__dict__["nodes"]
        weighted_edges = []
        edges = permutations(temp_dict.keys(), 2)
        for edge in edges:
            score = -self.teta_vec * self.phi(temp_dict[edge[0]], temp_dict[edge[1]])
            weighted_edges.append((temp_dict[edge[0]], temp_dict[edge[1]], score))
        return weighted_edges





def get_dicts(corpus):
    word_set = set()
    pos_set = set()
    for sentence in corpus:
        temp_dict = sentence.__dict__['nodes']
        for key in temp_dict.keys():
            word_set.add(temp_dict[key]['word'])
            pos_set.add(temp_dict[key]['ctag'])

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

def get_edges(t):
    """
     This function gets a tree and return all edges.
     """
    edges = []
    temp_dict = t.__dict__['nodes']
    for key in temp_dict.keys():
        head = temp_dict[key]['head']
        if head:
            edges.append((head, key))
    return edges




if __name__ == "__main__":
    corpus = dependency_treebank.parsed_sents()
    train_set, test_set = corpus[:int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]

    print((corpus[0]))

    # word_dict, pos_dict = get_dicts(corpus)
    # print(len(word_dict))
    # print(len(pos_dict))








    # 1. Implement Phi: Input: S sentence, u word, v word. Output: Vector F, shape=?
    #
    # 2. Implement Perceptron:
    #     w = vector(same shape of F)
    #     W_avg = w
    #     repeate 2 times:
    #         for every sentence S, RealTree T:
    #             E = {}
    #             for every u,v in S:
    #                 p = w * Phi(S, u, v)
    #                 add (u,v) with score p to E
    #             T_tag = MST algorithm(E) (Dont forget to negate the scores p -> -p)
    #             T_sum = Sum(Phi(S, e)) for e in T
    #             T_tag_sum = Sum(Phi(S, e)) for e in T_tag
    #             w = w + eta*(T_sum - T_tag_sum)
    #             W += w
    #     return W / (2 * # of sentences)
    #
    # 3. Evaluate:
    #     Define accuaracy as: shared edges from real graph and predicted graph, divided by # of words in the sentence
    #     Compute accuracy over all test set



    # concerns:
    # root node doesn't have a word.
