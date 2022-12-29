import nltk
from nltk import DependencyGraph

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank


def create_bi_dict(corpus):
    word_set = set()
    pos_set = set()
    for sentence in corpus:
        splited = sentence.to_conll(3).split()
        words = splited[0::3]
        poses = splited[1::3]
        for word in words:
            word_set.add(word)
        for pos in poses:
            pos_set.add(pos)
    #    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!              TO DO           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # create  a dict from the sets  for hashing words to index in vec


def phi(s, u, v):
    pass

if __name__ == "__main__":
    corpus = dependency_treebank.parsed_sents()
    train_set, test_set = corpus[:int(0.9 * len(corpus))], corpus[int(0.9 * len(corpus)):]

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
