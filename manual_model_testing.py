from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from keras.models import load_model

model = load_model('my_model.h5')
tokenizer = pickle.load(open('tokenizer', 'rb'))


def predict(input):
    word_sequences = tokenizer.texts_to_sequences([input])
    # word_sequences.append(w_seq)
    train_matrix = pad_sequences(
        word_sequences, maxlen=300)
    data = np.reshape(train_matrix[0], (1, -1))
    return model.predict_classes(data)
