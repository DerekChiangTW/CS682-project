import csv
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
import numpy as np


s_len = []
index = []
i = 0
with open('review_big.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sentence_review = sent_tokenize(row['text'])
        s_len.append(len(sentence_review))
        index.append(i)
        i += 1

# for key in star_len.keys():
# with open('pos_star_sentence_count', 'wb') as outfile:
#     pickle.dump(star_len, outfile)
# key = '5'

print(np.mean(s_len))
print(np.median(s_len))
plt.ylabel('Occurrence')
plt.xlabel('Number of Sentence in a paragraph')
plt.xlim((0,50))
# plt.scatter(index,s_len)
plt.hist(s_len)
plt.show()