# This file is for the sentiment analysis for the reviews for merchandise data, for clothing, jwellery and gifts dataset.

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from wordcloud import WordCloud, STOPWORDS

con = sqlite3.connect('merchandise_dataset.db')
stopwords = set(STOPWORDS)    

def cleanReviews(text):
	regEx = re.compile('[^a-z]+')
	text = text.lower()
	text = regEx.sub(' ', text).strip()
	return text

def bool_to_int(x):
    if x == 'negative':
        return 0
    return 1

def show_wc(data, title = None):
    wc = WordCloud(background_color='white', stopwords=stopwords, max_words=200, max_font_size=40, scale=3, random_state=1).generate(str(data))
    figure = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    if title: 
        figure.suptitle(title, fontsize=20)
        figure.subplots_adjust(top=2.3)
    plt.imshow(wc)
    plt.show()

def getSentiment(countVector, tfidf_transformer, model, review):
    transformed_count = countVector.transform([review])
    transformed_tf_idf = tfidf_transformer.transform(transformed_count)
    result = model.predict(transformed_tf_idf)[0]
    prob = model.predict_proba(transformed_tf_idf)[0]
    print("Sample estimated as %s: negative probability %f, positive probability %f" % (result.upper(), prob[0], prob[1]))

reviews = pd.read_sql_query(""" SELECT overall, summary, helpful, total FROM amazonReviews WHERE overall != 3""", con)

# Set the sentiment
reviews["sentiment"] = reviews["overall"].apply(lambda score: "positive" if score > 3 else "negative")
# Set the usefulness score
reviews["usefulScore"] = (reviews["helpful"]/reviews["total"]).apply(lambda n: "useful" if n > 0.8 else "useless")

# Clean the review texts
reviews["summaryClean"] = reviews["summary"].apply(cleanReviews)

# 80% train and 20% test
train, test = train_test_split(reviews, test_size=0.2)

# Get the frequency of each word ngram model
countVector = CountVectorizer(min_df = 1, ngram_range = (1, 4))
X_train_counts = countVector.fit_transform(train["summaryClean"])

#applying tf-idf to the count vector model
tfidf_transformer = TfidfTransformer()
X_train = tfidf_transformer.fit_transform(X_train_counts)

X_test_vector = countVector.transform(test["summaryClean"])
X_test = tfidf_transformer.transform(X_test_vector)

y_train = train["sentiment"]
y_test = test["sentiment"]

pred = dict()

mpl.rcParams['font.size']=12
mpl.rcParams['savefig.dpi']=100
mpl.rcParams['figure.subplot.bottom']=.1

show_wc(reviews["summaryClean"])
show_wc(reviews[reviews.overall == 1]["summaryClean"]) # low scored
show_wc(reviews[reviews.overall == 5]["summaryClean"]) # high scored
show_wc(reviews[reviews.overall == 2]["summaryClean"]) # average scored

model = MultinomialNB().fit(X_train, y_train)
pred['Multinomial'] = model.predict(X_test)

model = BernoulliNB().fit(X_train, y_train)
pred['Bernoulli'] = model.predict(X_test)

l_reg = LogisticRegression(C=1e5)
l_reg_result = l_reg.fit(X_train, y_train)
pred['Logistic'] = l_reg.predict(X_test)

vfunc = np.vectorize(bool_to_int)

idx = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in pred.items():
    fp_rate, tp_rate, thresholds = roc_curve(y_test.map(bool_to_int), vfunc(predicted))
    roc_auc = auc(fp_rate, tp_rate)
    plt.plot(fp_rate, tp_rate, colors[idx], label='%s: AUC %0.2f'% (model,roc_auc))
    idx += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print(metrics.classification_report(y_test, pred['Logistic'], target_names = ["positive", "negative"]))

'''
             precision    recall  f1-score   support

   positive       0.78      0.70      0.74      5291
   negative       0.97      0.98      0.97     44360

avg / total       0.94      0.95      0.95     49651
'''
print(accuracy_score(y_test, pred['Bernoulli'])) # 0.897383738495
print(accuracy_score(y_test, pred['Multinomial'])) # 0.915188012326
print(accuracy_score(y_test, pred['Logistic'])) # 0.946708021994

features = countVector.get_feature_names()
feature_coefs = pd.DataFrame( data = list(zip(features, l_reg_result.coef_[0])), columns = ['feature', 'coefficient'] )

print(feature_coefs.sort_values(by='coefficient')) # [537027 rows x 2 columns]
getSentiment(countVector, tfidf_transformer, l_reg, "Heavenly Highway Hymns")
# Sample estimated as POSITIVE: negative probability 0.001339, positive probability 0.998661
getSentiment(countVector, tfidf_transformer, l_reg, "Very oily and creamy. Not at all what I expected... it just looked awful!!! Plus, took FOREVER to arrive.")
# Sample estimated as NEGATIVE: negative probability 0.997464, positive probability 0.002536
getSentiment(countVector, tfidf_transformer, l_reg, "Weird smelling shampoo!.")
# Sample estimated as NEGATIVE: negative probability 0.859040, positive probability 0.140960
con.close()