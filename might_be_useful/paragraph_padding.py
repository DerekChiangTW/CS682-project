import numpy as np


def paragraph_padding(matrix, max_sentence):
    """
    Input is a 2D numpy array and max_sentence length, return a padded 2D numpy array which has shape(max_sentence,col)
    """
    row = matrix.shape[0]
    col = matrix.shape[1]
    new_matrix = np.zeros((max_sentence, col))

    for i in range(row):
        if i >= max_sentence:
            break
        else:
            new_matrix[i] = matrix[i]

    return new_matrix
