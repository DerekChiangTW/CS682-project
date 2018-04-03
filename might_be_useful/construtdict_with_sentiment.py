import csv
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from collections import defaultdict
from keras.preprocessing.text import Tokenizer


def get_sentiment_info(sent_analyzer, review):

    sentiment_stat = [0] * 5
    ss = sent_analyzer.polarity_scores(review)
    sentiment_stat[0] = ss['neg']
    sentiment_stat[1] = ss['pos']
    sentiment_stat[2] = ss['neu']
    sentiment_stat[3] = ss['compound']
    sentiment_stat[4] = ss['pos'] - ss['neg']
    # Sentiment percentage
    bow = review.split()
    sentiment_percentage = [0] * 3  # 0 for pos, 1 for neg, 2 for neu
    for word in bow:
        word_sentiment = sid.polarity_scores(word)
        if word_sentiment['pos'] > word_sentiment['neg'] and word_sentiment['pos'] > word_sentiment['neu']:
            sentiment_percentage[0] += 1
        elif word_sentiment['neg'] > word_sentiment['pos'] and word_sentiment['neg'] > word_sentiment['neu']:
            sentiment_percentage[1] += 1
        else:
            sentiment_percentage[2] += 1
    sentiment_percentage = (np.array(sentiment_percentage) / sum(sentiment_percentage)).tolist()
    sentiment_stat = sentiment_stat + sentiment_percentage
    return sentiment_stat


if __name__ == '__main__':

    word2index = defaultdict(int)
    index2word = defaultdict(str)
    bow = defaultdict(float)
    MAXWORD = 8000
    stopWords = set(stopwords.words('english'))
    review_count = 1
    reviews = []
    labels = []
    reviews_sentiment = []
    sid = SentimentIntensityAnalyzer()

    with open('review_big_1207.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print("Processed ", review_count, " reviews")
            review_count += 1
            text = row['text']
            labels.append(row['stars'])
            reviews.append(text)
            reviews_sentiment.append(get_sentiment_info(sid, text))

    assert len(reviews) == 50000

    # transform text to sequence
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    data = pad_sequences(sequences, maxlen=300)
    labels = np.reshape(labels, (-1, 1))
    reviews_sentiment = np.array(reviews_sentiment)
    print(labels.shape)
    print(reviews_sentiment.shape)

    with open('train_matrix', 'wb') as f:
        pickle.dump(data, f)
        print("train matrix saved!!")

    with open('train_label', 'wb') as f:
        pickle.dump(labels, f)
        print("train label saved!!")

    with open('train_sentiment', 'wb') as f:
        pickle.dump(reviews_sentiment, f)
        print("train sentiment saved!!")
