import csv
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from collections import defaultdict
from keras.preprocessing.text import Tokenizer

word2index = defaultdict(int)
index2word = defaultdict(str)
bow = defaultdict(float)
MAXWORD = 8000
stopWords = set(stopwords.words('english'))
current_index = 1

reviews = []
labels = []
with open('review_big.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        labels.append(row['stars'])
        reviews.append(text)

assert len(reviews) == 50000
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
data = pad_sequences(sequences, maxlen=300)
labels = np.reshape(labels, (-1, 1))

pickle.dump(tokenizer, open('tokenizer', 'wb'))

with open('train_matrix', 'wb') as f:
    pickle.dump(data, f)

with open('train_label', 'wb') as f:
    pickle.dump(labels, f)
