import csv
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.stem.porter import *
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import sparse
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv, re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import sparse
import string
import copy

def hp_vs_score_plot(x_array, y_array_1, y_array_2, clf_name, hp_name):  # for plotting the hp vs rae for each model

    #pic_name = './' + clf_name + '_hp_vs_score.eps'
    pic_name = './' + clf_name + '_hp_vs_score.png'
    title_name = "Train Score and Test Score for each " + hp_name + " of " + clf_name

    # Plot a line graph
    labels = ["Test", "Train"]
    index = x_array

    plt.figure(2, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    plt.plot(index, y_array_1, 'or-', linewidth=3)  # Plot the first series in red with circle marker
    plt.plot(index, y_array_2, 'sb-', linewidth=3)  # Plot the first series in blue with square marker

    # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel("Score")  # Y-axis label
    plt.xlabel(hp_name)  # X-axis label
    plt.title(title_name)  # Plot title
    plt.legend(labels, loc="best")
    plt.savefig(pic_name)
    #plt.savefig(pic_name, format='eps', dpi=300)
    plt.close()

def append_sentiment_info(data_sparse_matrix, original_data):
    sid = SentimentIntensityAnalyzer()
    # Appending sentiment to BOW
    sentiment_stat = []
    for review in original_data:
        sentence_review = sent_tokenize(review)
        neg_sum = 0
        pos_sum = 0
        neu_sum = 0
        compound_sum = 0
        diff_sum = 0
        for word in sentence_review:
            ss = sid.polarity_scores(word)
            diff = ss['pos'] - ss['neg']
            neg_sum += ss['neg']
            pos_sum += ss['pos']
            neu_sum += ss['neu']
            compound_sum += ss['compound']
            diff_sum += diff
        sentiment_stat.append([neg_sum, pos_sum, neu_sum, compound_sum, diff_sum])
    sentiment_stat = np.array(sentiment_stat)
    data_sparse_matrix = data_sparse_matrix.toarray()
    data_sparse_matrix = np.append(data_sparse_matrix, sentiment_stat, axis=1)
    data_sparse_matrix = np.matrix(data_sparse_matrix)
    return sparse.csr_matrix(data_sparse_matrix)

def get_data(data_limit = 40000):
    rows = []
    row_count = 0
    with open('./dataset/parsed_review.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row_count > data_limit:
                break
            rows.append(row)
            row_count += 1
    csvfile.close()
    rows = shuffle(rows, random_state=7)
    X_train, y_train, X_test, y_test = [], [], [], []
    translator = str.maketrans('', '', string.punctuation)
    for i in range(len(rows)): # basic split with a little bit of pre-processing
        if i < int(0.8 * len(rows)):
            X_train.append(rows[i]['text'].replace('\'', ' ').translate(translator))
            y_train.append(rows[i]['stars'])
        else:
            X_test.append(rows[i]['text'].replace('\'', ' ').translate(translator))
            y_test.append(rows[i]['stars'])

    rows = None  # release the memory

    return (X_train, y_train, X_test, y_test)

def data_preprocessing(X_train, X_test, unigram = False, include_sentiment = False):

    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    count_vect = CountVectorizer(stop_words='english', analyzer=stemmed_words,
                                 ngram_range=(1, 1) if unigram else (1, 2))
    tf_transformer = TfidfTransformer(use_idf=True)

    for is_train in [True, False]: # train + tranform on the train data, and then tranform the test data
        # Bag of words
        data_backup_use, data_backup = None, None

        if is_train:
            data_backup_use = copy.copy(X_train)
            data_backup = copy.copy(X_train)
        else:
            data_backup_use = copy.copy(X_test)
            data_backup = copy.copy(X_test)

        train_counts = count_vect.fit_transform(data_backup_use)

        if is_train: # if training data, train the tf_transformer
            tf_transformer.fit(train_counts)

        word_count_feat = tf_transformer.transform(train_counts)
        # Adding sentiments
        if include_sentiment:
            data_backup_use = append_sentiment_info(data_backup, word_count_feat)
        else:
            data_backup_use = word_count_feat

        if is_train:  # if training data, train the tf_transformer
            X_train = data_backup_use
        else:
            X_test = data_backup_use

    return (X_train, X_test)

def train_classifier(clf, X_train, y_train, X_test, y_test):
    # Starting training
    for chosen_clf in clf_to_run:
        text_clf = Pipeline([
            ('clf', clf['classifier']),
        ])

        parameters = {}
        parameters[clf['hp_name']] = clf['hps']
        gs_clf = GridSearchCV(text_clf, parameters, verbose=10, n_jobs=6)
        gs_clf = gs_clf.fit(X_train, y_train)
        print(gs_clf.best_score_)
        print(gs_clf.best_estimator_)

        print(gs_clf.cv_results_['mean_test_score'].tolist())
        print(gs_clf.cv_results_['mean_train_score'].tolist())
        hp_vs_score_plot(clf['hps'],
                         gs_clf.cv_results_['mean_test_score'].tolist(),
                         gs_clf.cv_results_['mean_train_score'].tolist(), chosen_clf, clf['hp_name'])

        predicted = gs_clf.predict(X_test)
        print(metrics.classification_report(y_test, predicted))

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data( data_limit=1000)
    X_train, X_test = data_preprocessing(X_train, X_test)

    clf_NB, clf_RF, clf_LR = {}, {}, {}
    clfs = [clf_NB, clf_RF, clf_LR]

    clf_NB['name'] = "Naive Bayes"
    clf_NB['classifier'] = BernoulliNB()
    clf_NB['hp_name'] = 'clf__alpha'
    clf_NB['hps'] = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2]

    clf_RF['name'] = "Random Forest"
    clf_RF['classifier'] = RandomForestClassifier()
    clf_RF['hp_name'] = 'clf__n_estimators'
    clf_RF['hps'] = [50, 100, 150, 200, 250, 300]

    clf_LR['name'] = "Logistic Regression"
    clf_LR['classifier'] = LogisticRegression()
    clf_LR['hp_name'] = 'clf__C'
    clf_LR['hps'] = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6]

    for clf in clfs:
        train_classifier(clf, X_train, y_train, X_test, y_test)

