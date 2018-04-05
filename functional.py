import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Embedding, Dropout, Input, Conv1D, MaxPooling1D, concatenate, add, \
    BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.backend import shape
import numpy as np

################################################################
# ***********************DATA LOADING****************************
################################################################
NUMOFWORDS = 300
TOTALWORDS = 20000
EMBEDDING_DIM = 128

reviews = []
labels = []
reviews_sentiment = []

with open('./dataset/parsed_review.csv', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        labels.append(row['stars'])
        reviews.append(text)
csvfile.close()
print("******finished loading reviews******")

with open('./dataset/parsed_review_sentiment.csv', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        temp = [row['0'], row['1'], row['2'], row['3'], row['4']]
        reviews_sentiment.append(temp)
csvfile.close()
print("******finished loading reviews sentiment******")

################################################################
# *******************DATA PREPROCESSING**************************
################################################################
assert len(reviews) == 50000
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
training_data = pad_sequences(sequences, maxlen=300)
true_label = np.reshape(labels, (-1, 1))
training_data_sentiment = np.array(reviews_sentiment)
print('Training data shape is {}'.format(training_data.shape))
print('Training Label shape is {}'.format(true_label.shape))

################################################################
# ******************* ARCHITECTURE**************************
################################################################
# main branch
"""
Input: (batch, sequence)
Embedding: (batch, sequence, embedding)
Conv1D: (batch, new_steps, filters)
MaxPooling1D: (batch_size, downsampled_steps, features)

"""

main_input = Input(shape=(300,), dtype='int32', name='main_input')  # input for the main branch: review
x = Embedding(output_dim=EMBEDDING_DIM, input_dim=TOTALWORDS, input_length=NUMOFWORDS)(main_input)  # 3D:(Batch, Sequence, Embedding)
x = Dropout(rate=0.25)(x)
x = Conv1D(64, 5, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)  # sentense embedding
lstm_out = LSTM(128)(x)  # paragraph embedding
auxiliary_output = Dense(6, activation='softmax', name='aux_output')(lstm_out)  # loss function for paragraph

# extra branch: sentiment
auxiliary_input = Input(shape=(5,), name='aux_input')  # input for the side branch: sentiment
print(shape(auxiliary_input))
auxiliary_input_dense = Dense(12, activation='relu')(auxiliary_input)
auxiliary_input_dense = BatchNormalization()(auxiliary_input_dense)
auxiliary_input_dense = Dense(8, activation='relu')(auxiliary_input_dense)
auxiliary_input_dense = BatchNormalization()(auxiliary_input_dense)
auxiliary_input_dense = Dense(6, activation='softmax', name='sent_side_output')(auxiliary_input_dense)
print(shape(auxiliary_input))

# Two branches merge
x = concatenate([auxiliary_output, auxiliary_input_dense])
x = Dense(16, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(10, activation='relu')(x)
x = BatchNormalization()(x)
main_output = Dense(6, activation='softmax', name='main_output')(x)

################################################################
# ******************* ARCHITECTURE ENDED************************
################################################################

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output, auxiliary_input_dense])

model.compile(optimizer='adam',
              loss={'main_output': 'sparse_categorical_crossentropy', 'aux_output': 'sparse_categorical_crossentropy',
                    'sent_side_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'],
              loss_weights={'main_output': 0.6, 'aux_output': 0.25, 'sent_side_output': 0.15})

earlystop = EarlyStopping(monitor='val_main_output_acc',min_delta=0.05, patience=3)


print("start training")
history = model.fit({'main_input': training_data, 'aux_input': training_data_sentiment},
                    {'main_output': true_label, 'aux_output': true_label,
                     'sent_side_output': true_label},
                    epochs=10, callbacks=[earlystop], validation_split=0.2)

model.save('my_model.h5')
