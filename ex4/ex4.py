import nltk
from itertools import product
from nltk import DependencyGraph

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank


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

    def forward(self, text):
        pass
    def predict(self, text):
        pass

    def get_feature_index(self, u, v, is_word):
        if is_word:
            return self.words_dict[u] * self.words_dict_size + self.words_dict[v]
        return self.words_dict_size ** 2 + self.pos_dict[u] * self.pos_dict_size + self.pos_dict[v]

    def phi(self, s, u, v):
        pass

def get_dicts(corpus):
    word_set = set()
    pos_set = set()
    words_dict = dict()
    pos_dict = dict()
    for sentence in corpus:
        splited = sentence.to_conll(3).split()
        words = splited[0::3]
        poses = splited[1::3]
        for word in words:
            word_set.add(word)
        for pos in poses:
            pos_set.add(pos)
    i = 0
    for word in word_set:
        words_dict[word] = i
        i += 1
    i = 0
    for pos in pos_dict:
        pos_dict[pos] = i
        i += 1
    return words_dict, pos_dict

if __name__ == "__main__":
    corpus = dependency_treebank.parsed_sents()
    train_set, test_set = corpus[:int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]

    # word_dict, pos_dict = get_dicts(corpus)


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
