import nltk

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
import numpy as np


class MSTparser():
    """
    An MSTparser.
    """
    def __init__(self, corpus, words_dic, pos_dict):
        self.corpus = corpus
        self.words_dict = words_dic
        self.pos_dict = pos_dict
        self.words_dict_size = len(words_dic)
        self.pos_dict_size = len(pos_dict)
        self.vec_dim = self.words_dict_size ** 2 + self.pos_dict_size ** 2

    def forward(self, text):
        pass
    def predict(self, text):
        pass

    def get_feature_index(self, u, v, is_word):
        if is_word:
            return self.words_dict[u] * self.words_dict_size + self.words_dict[v]
        return self.words_dict_size ** 2 + self.pos_dict[u] * self.pos_dict_size + self.pos_dict[v]

    def phi(self, s, u, v):
        vec = np.zeros(self.vec_dim, dtype=bool)



def get_dicts(corpus):
    word_set = set()
    pos_set = set()
    words_dict = dict()
    pos_dict = dict()
    for sentence in corpus:
        temp_dict = sentence.__dict__['nodes']
        for key in temp_dict.keys():
            word_set.add(temp_dict[key]['word'])
            pos_set.add(temp_dict[key]['ctag'])
    i = 0
    for word in word_set:
        words_dict[word] = i
        i += 1
    i = 0
    for pos in pos_set:
        pos_dict[pos] = i
        i += 1
    return words_dict, pos_dict

if __name__ == "__main__":
    corpus = dependency_treebank.parsed_sents()
    train_set, test_set = corpus[:int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]

    word_dict, pos_dict = get_dicts(corpus)
    print(len(word_dict))
    print(len(pos_dict))
    # dict = corpus[0].__dict__['nodes']
    # print(dict)
    # for key in dict.keys():
    #     print(dict[key]['word'])
    # print(corpus[0].__dict__['nodes'][0])


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
