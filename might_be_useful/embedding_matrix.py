import numpy as np
from gensim.models.keyedvectors import KeyedVectors

def get_embedding_matrix(bow, word2index, word2vec, embedding_length, filter_threshold):
    number_unique_vocab = len(bow)+1  # give a bound to number of vocab
    embedding_matrix = np.zeros(
        (number_unique_vocab, embedding_length))  # embedding matrix where each row corresponding to one vector
    for word, count in bow.items():
        word_index = word2index[word]
        if word in word2vec:
            if count >= filter_threshold:  # only add the word vector if word count is greater than threshold
                word_vec = word2vec[word]
                embedding_matrix[word_index] = word_vec

    return embedding_matrix


def make_embedding_pickle():

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        # print(word)
    f.close()

    with open('embeddings.pickle', 'wb') as handle:
        pickle.dump(embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    import os
    import pickle

    BASE_DIR = ''
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 100
    word2index = pickle.load(open('word2index', 'rb'))
    bow = pickle.load(open('bow', 'rb'))
    index2word = pickle.load(open('index2word', 'rb'))

    print('Indexing word vectors.')
    word_vectors = KeyedVectors.load_word2vec_format('glove.6B/glove.6B.100d.w2v', binary=False)
    # embeddings_index = pickle.load(open('embeddings.pickle', 'rb'))

    print('creating embedding matrix')
    embedding_matrix = get_embedding_matrix(bow, word2index, word_vectors, EMBEDDING_DIM, 50)

    print(index2word[78])
    print(embedding_matrix[78])
    pickle.dump(embedding_matrix,open('embedding_matrix', 'wb'))
    # print(np.array_equal(embedding_matrix[1], word_vectors['people']))
    # print(np.array_equal(embedding_matrix[3], word_vectors['year']))
