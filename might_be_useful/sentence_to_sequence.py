import numpy as np
from paragraph_padding import paragraph_padding
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.sequence import pad_sequences
import pickle
import csv
#################################################
'STEP1 - Reading data and necessary dictionarys'

word2index = pickle.load(open('word2index','rb'))
bow = pickle.load(open('bow','rb'))
index2word = pickle.load(open('index2word','rb'))

#################################################
'STEP2 - Implementing Sentence to sequence'
NUMOFREVIEWS = 40000
NUMOFSEN = 10
NUMOFWORDS = 60

index = 0
label = []
word_sequences = []

with open('review_big.csv') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        if index >= NUMOFREVIEWS:
            print('Reach the MAX Review done reading')
            break

        label.append(row['stars'])
        words = word_tokenize(row['text'])
        w_seq = []
        for word in words:
            lw = word.lower()
            w_seq.append(word2index[lw])

        word_sequences.append(w_seq)

        #Start pading each sentence
        index = index + 1


train_matrix = pad_sequences(word_sequences,maxlen=1000,padding='post',truncating='post')
print(train_matrix[0])
label = np.reshape(label, (-1, 1))
#################################################
'STEP3 - Saving training matrix'
with open('train_matrix','wb') as f:
    pickle.dump(train_matrix,f)

with open('train_label','wb') as f:
    pickle.dump(label,f)