import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Dense, Embedding, Dropout, Input, Conv1D, MaxPooling1D, concatenate, add
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.backend import shape
import numpy as np

NUMOFWORDS = 300
TOTALWORDS = 20000
EMBEDDING_DIM = 128


def get_sentiment_info(review):
    sid = SentimentIntensityAnalyzer()
    sentiment_stat = [0] * 5
    ss = sid.polarity_scores(review)
    sentiment_stat[0] = ss['neg']
    sentiment_stat[1] = ss['pos']
    sentiment_stat[2] = ss['neu']
    sentiment_stat[3] = ss['compound']
    sentiment_stat[4] = ss['pos'] - ss['neg']
    return sentiment_stat


reviews = []
labels = []
reviews_sentiment = []
sid = SentimentIntensityAnalyzer()
with open('review_big_1207.csv', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        labels.append(row['stars'])
        reviews.append(text)
        reviews_sentiment.append(get_sentiment_info(text))


assert len(reviews) == 50000
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
train = pad_sequences(sequences, maxlen=300)
label = np.reshape(labels, (-1, 1))
train_sentiment = np.array(reviews_sentiment)

print('Training data shape is {}'.format(train.shape))
print('Training Label shape is {}'.format(label.shape))


# define input
main_input = Input(shape=(300,), dtype='int32', name='main_input')
# define embedding
x = Embedding(output_dim=EMBEDDING_DIM, input_dim=TOTALWORDS,
              input_length=NUMOFWORDS)(main_input)
x = Dropout(rate=0.25)(x)
x = Conv1D(64, 5, activation='relu')(x)
# sentense embedding
x = MaxPooling1D(pool_size=2)(x)
# paragraph embedding
lstm_out = LSTM(128)(x)
# loss function for paragraph
auxiliary_output = Dense(6, activation='softmax', name='aux_output')(lstm_out)
# extra branch: sentiment
auxiliary_input = Input(shape=(8,), name='aux_input')
print(shape(auxiliary_input))

# We stack a deep densely-connected network on top
auxiliary_input_dense = Dense(64, activation='sigmoid')(auxiliary_input)
auxiliary_input_dense = Dense(64, activation='sigmoid')(auxiliary_input_dense)
auxiliary_input_dense = Dense(
    6, activation='softmax', name='sent_side_output')(auxiliary_input_dense)
print(shape(auxiliary_input))
x = add([auxiliary_output, auxiliary_input_dense])


# main output

# We stack a deep densely-connected network on top
x = Dense(64, activation='softmax')(x)
#x = Dense(64, activation='relu')(x)
#x = Dense(64, activation='relu')(x)

main_output = Dense(6, activation='softmax', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[
              main_output, auxiliary_output, auxiliary_input_dense])

model.compile(optimizer='rmsprop',
              loss={'main_output': 'sparse_categorical_crossentropy', 'aux_output': 'sparse_categorical_crossentropy',
                    'sent_side_output': 'sparse_categorical_crossentropy'},
              metrics=['accuracy'],
              loss_weights={'main_output': 1., 'aux_output': 0.2, 'sent_side_output': 0.7})

earlystop = EarlyStopping(monitor='val_main_output_acc',
                          min_delta=0.05, patience=3)

history = model.fit({'main_input': train, 'aux_input': train_sentiment},
                    {'main_output': label, 'aux_output': label,
                        'sent_side_output': label},
                    epochs=15, callbacks=[earlystop], validation_split=0.2)

model.save('my_model.h5')
