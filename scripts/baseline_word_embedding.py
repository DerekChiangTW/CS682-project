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
from keras.models import load_model
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
################################################################
# ***********************DATA LOADING****************************
################################################################
NUMOFWORDS = 300
TOTALWORDS = 20000
EMBEDDING_DIM = 128

reviews = []
labels = []
filepath_train = __file__[:-34] + '/dataset/train_review.csv'
filepath_validate = __file__[:-34] + '/dataset/validate_review.csv'
#filepath_train = __file__[:-34] + '/dataset/parsed_review.csv'
with open(filepath_train, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        labels.append(row['stars'])
        reviews.append(text)
csvfile.close()
with open(filepath_validate, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        labels.append(row['stars'])
        reviews.append(text)
csvfile.close()
print("******finished loading reviews******")
reviews_test = []
labels_test = []
filepath_train = __file__[:-34] + '/dataset/test_review.csv'
with open(filepath_train, encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        text = row['text']
        labels_test.append(int(row['stars']))  # CAST TO INT
        reviews_test.append(text)
csvfile.close()
print("******finished loading reviews******")

################################################################
# *******************DATA PREPROCESSING**************************
################################################################
#assert len(reviews) == 50000
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
training_data = pad_sequences(sequences, maxlen=300)
true_label = np.reshape(labels, (-1, 1))
print('Training data shape is {}'.format(training_data.shape))
print('Training Label shape is {}'.format(true_label.shape))

sequences = tokenizer.texts_to_sequences(reviews_test)
testing_data = pad_sequences(sequences, maxlen=300)
true_label_test = np.reshape(labels_test, (-1, 1))
print('Testing data shape is {}'.format(testing_data.shape))
print('Testing Label shape is {}'.format(true_label_test.shape))
train = True
if train:
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
    x = LSTM(128)(x)  # paragraph embedding
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    main_output = Dense(6, activation='softmax', name='main_output')(x)

    ################################################################
    # ******************* ARCHITECTURE ENDED************************
    ################################################################
    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam',
                  loss={'main_output': 'sparse_categorical_crossentropy'},
                  metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_acc',min_delta=0.05, patience=3)

    print("start training")
    history = model.fit({'main_input': training_data},
                        {'main_output': true_label},
                        epochs=10, callbacks=[earlystop], validation_split=0.2)
    model.save('my_model.h5')

model = load_model('my_model.h5')
prediction_test = model.predict(testing_data)
prediction_test[:,0] = np.zeros(testing_data.shape[0])
#prediction_test = np.delete(prediction_test, 5, 1)
print (prediction_test.shape)
predicted_label = np.argmax(prediction_test,axis = 1).flatten()

max = 0
min = 999
for i in range(testing_data.shape[0]):
    #predicted_label[i] += 1
    if predicted_label[i] >= max:
        max = predicted_label[i]
    if predicted_label[i] <=min:
        min = predicted_label[i]

print(max)
print(min)

print("Test Accuracy: ", accuracy_score(true_label_test, predicted_label))
print("Test Precision: ", precision_score(true_label_test, predicted_label,average=None))
print("Test Recall: ", recall_score(true_label_test, predicted_label,average=None))
print("Test F1-Score: ", f1_score(true_label_test, predicted_label,average=None))
