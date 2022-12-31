import nltk

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
    def __init__(self, words_dic, pos_dict, n_iterations):
        self.words_dict = words_dic
        self.pos_dict = pos_dict
        self.words_dict_size = len(words_dic)
        self.pos_dict_size = len(pos_dict)
        self.vec_dim = self.words_dict_size ** 2 + self.pos_dict_size ** 2
        self.teta_vec = np.zeros(self.vec_dim, dtype=float) # dtype = float ???????????????
        self.acumulative_teta = np.zeros(self.vec_dim, dtype=float)
        self.n_iteration = n_iterations
        self.mode_is_train = True


    def forward(self, t):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!        in forward         !!!!!!!!!!!!!!!!!!!")
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
        return accuracy / len(t.nodes)

    def train_model(self, train_set):
        size = len(train_set)
        for i in range(self.n_iteration):
            shuff = np.random.permutation(size)
            for s in shuff:
                self.forward(train_set[s])
        self.acumulative_teta /= (self.n_iteration/size)

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

    def phi(self, u, v):
        words_index = self.get_feature_index(u['word'], v['word'], is_word=True)
        pos_index = self.get_feature_index(u['tag'], v['tag'], is_word=False)
        if self.mode_is_train:
            return self.teta_vec[words_index] + self.teta_vec[pos_index]
        else:
            return self.acumulative_teta[words_index] + self.acumulative_teta[pos_index]

    def get_vec_score_of_t(self, t, arcs_gold,  arcs_max):
        result = {}
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
        return result

    def get_all_possible_arcs(self, t):
        """
            get all possible edges to be sent to chu lie algorithm
        """
        weighted_arcs = []
        arcs = permutations(range(len(t.nodes)), 2)
        for arc in arcs:
            score = -self.phi(t.nodes[arc[0]], t.nodes[arc[1]])
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
        if head:
            arcs.append(Arc(head, i, 0))
    return arcs




if __name__ == "__main__":
    corpus = dependency_treebank.parsed_sents()
    train_set, test_set = corpus[:int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]


    word_dict, pos_dict = get_dicts(corpus)
    print(len(corpus))
    model = MSTparser(word_dict,pos_dict, 2)
    model.train_model(train_set[:1500])
    print(model.test_model(test_set))









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
