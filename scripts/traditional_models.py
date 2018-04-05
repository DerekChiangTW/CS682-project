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

def hp_vs_score_plot(x_array, y_array_1, y_array_2, clf_name, hp_name):  # for plotting the hp vs rae for each model

    #pic_name = './' + clf_name + '_hp_vs_score.eps'
    pic_name = './' + clf_name + '_hp_vs_score.png'
    title_name = "Train Score and Test Score for each " + hp_name + " of " + clf_name

    """
    for i in range(0, len(y_array_1)):
        y_array_1[i] *= -1
        y_array_2[i] *= -1
    """
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


def append_sentiment_info(original_data, data_sparse_matrix):
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


def get_features_count_vec():
    threshold = 40000
    unigram = False
    include_sentiment = True
    clf_to_run = ['NB','LR', 'RF']
    rows = []
    row_count = 0
    with open('review_big.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row_count <= threshold:
                rows.append(row)
            row_count += 1
    rows = shuffle(rows, random_state=7)
    separated_data = [[], [], [], []]
    translator = str.maketrans('', '', string.punctuation)
    for i in range(len(rows)):
        if i < int(0.8 * len(rows)):
            separated_data[0].append(rows[i]['text'].replace('\'', ' ').translate(translator))
            separated_data[1].append(rows[i]['stars'])
        else:
            separated_data[2].append(rows[i]['text'].replace('\'', ' ').translate(translator))
            separated_data[3].append(rows[i]['stars'])

    rows = None  # release the memory


    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()
    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))
    count_vect = CountVectorizer(stop_words='english', analyzer=stemmed_words,
                                 ngram_range=(1, 1) if unigram else (1, 2))
    tf_transformer = TfidfTransformer(use_idf=True)
    # ********************Data Processing**************************
    for i in [0, 2]:
        # BOW
        data_backup = separated_data[i]
        if i == 0:
            train_counts = count_vect.fit_transform(separated_data[i])
            tf_transformer.fit(train_counts)
        else:
            train_counts = count_vect.transform(separated_data[i])
        word_count_feat = tf_transformer.transform(train_counts)
        # Adding sentiments
        if include_sentiment:
            separated_data[i] = append_sentiment_info(data_backup, word_count_feat)
        else:
            separated_data[i] = word_count_feat
    # ************************************************************************

    # To add a classifier, please update the following: hp_names, hp_param_names, hp, clf_lists
    hp_names = {}
    hp_names['NB'] = 'alpha'
    hp_names['RF'] = 'n_estimators'
    hp_names['LR'] = 'C'

    hp_param_names = {}
    hp_param_names['NB'] = 'clf__alpha'
    hp_param_names['RF'] = 'clf__n_estimators'
    hp_param_names['LR'] = 'clf__C'

    hp = {}
    hp['NB'] = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2]
    hp['RF'] = [50, 100, 150, 200, 250, 300]
    hp['LR'] = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6]

    clf_lists = {}
    clf_lists['NB'] = BernoulliNB()
    clf_lists['RF'] = RandomForestClassifier()
    clf_lists['LR'] = LogisticRegression()

    # Starting training
    for chosen_clf in clf_to_run:

        text_clf = Pipeline([
            ('clf', clf_lists[chosen_clf]),
        ])

        parameters = {}
        parameters[hp_param_names[chosen_clf]] = hp[chosen_clf]
        gs_clf = GridSearchCV(text_clf, parameters, verbose=10, n_jobs=6)
        gs_clf = gs_clf.fit(separated_data[0], separated_data[1])
        print(gs_clf.best_score_)
        print(gs_clf.best_estimator_)

        print(gs_clf.cv_results_['mean_test_score'].tolist())
        print(gs_clf.cv_results_['mean_train_score'].tolist())
        hp_vs_score_plot(hp[chosen_clf],
                         gs_clf.cv_results_['mean_test_score'].tolist(),
                         gs_clf.cv_results_['mean_train_score'].tolist(), chosen_clf, hp_names[chosen_clf])

        predicted = gs_clf.predict(separated_data[2])
        print(metrics.classification_report(separated_data[3], predicted))


if __name__ == '__main__':
    get_features_count_vec()
