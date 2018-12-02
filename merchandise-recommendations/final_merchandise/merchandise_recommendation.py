import numpy as np
import pandas as pd
import re

from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error

def cleanReviews(text):
	regEx = re.compile('[^a-z]+')
	text = text.lower()
	text = regEx.sub(' ', text).strip()
	return text

df = pd.read_csv("reviews_final.csv")

print(df.columns)
#print(df.shape)
count = df.groupby("asin", as_index=False).count()
dfMerged = pd.merge(df, count, how='right', on=['asin'])
dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
dfMerged["overallScore"] = dfMerged["overall_x"]
dfMerged["summaryReview"] = dfMerged["summary_x"]
dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 50]
#print(dfCount)

productReview = df.groupby("asin", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("asin")["summaryReview"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.to_csv("ProductReviewSummary.csv")

df_reviews_final = pd.read_csv("ProductReviewSummary.csv")
df_reviews_final = pd.merge(df_reviews_final, productReview, on="asin", how='inner')
df_reviews_final = df_reviews_final[['asin','summaryReview','overall']]

# clean the summary texts and remove the duplicates
df_reviews_final["summaryClean"] = df_reviews_final["summaryReview"].apply(cleanReviews)
df_reviews_final = df_reviews_final.drop_duplicates(['overall'], keep='last')
df_reviews_final = df_reviews_final.reset_index()

reviews = df_reviews_final["summaryClean"] 
tfIdfVector = TfidfVectorizer(max_features = 300, stop_words='english')
transformedReviews = tfIdfVector.fit_transform(reviews)

dfReviews = pd.DataFrame(transformedReviews.A, columns=tfIdfVector.get_feature_names())
dfReviews = dfReviews.astype(int)
#dfReviews.to_csv("dfReviews.csv")

X = np.array(dfReviews)
# create 9o% train and and 10% test
tpercent = 0.9
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
len_train = len(dfReviews_train)
len_test = len(dfReviews_test)

neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)


for i in range(len_test):
    res = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = res[1]

    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    #print ("Based on product reviews, for ", df_reviews_final["asin"][len_train + i] ," average rating is ",df_reviews_final["overall"][len_train + i])
    #print ("The first similar product is ", df_reviews_final["asin"][first_related_product] ," average rating is ",df_reviews_final["overall"][first_related_product])
    #print ("The second similar product is ", df_reviews_final["asin"][second_related_product] ," average rating is ",df_reviews_final["overall"][second_related_product])
    #print ("-----------------------------------------------------------")

df5_train_target = df_reviews_final["overall"][:len_train]
df5_test_target = df_reviews_final["overall"][len_train:len_train+len_test]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

k_neighbors = 3
knn_clf = neighbors.KNeighborsClassifier(k_neighbors, weights='distance')
knn_clf.fit(dfReviews_train, df5_train_target)
knn_pred_test = knn_clf.predict(dfReviews_test)

print(classification_report(df5_test_target, knn_pred_test))
'''
             precision    recall  f1-score   support

          3       0.00      0.00      0.00         2
          4       0.90      0.95      0.92        19

avg / total       0.81      0.86      0.84        21

'''

print (accuracy_score(df5_test_target, knn_pred_test)) # 0.857142857143

print(mean_squared_error(df5_test_target, knn_pred_test)) # 0.142857142857


df5_train_target = df_reviews_final["overall"][:len_train]
df5_test_target = df_reviews_final["overall"][len_train:len_train+len_test]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

k_neighbors = 5
knn_clf = neighbors.KNeighborsClassifier(k_neighbors, weights='distance')
knn_clf.fit(dfReviews_train, df5_train_target)
knn_pred_test = knn_clf.predict(dfReviews_test)
#print (knn_pred_test)

print(classification_report(df5_test_target, knn_pred_test))

'''
             precision    recall  f1-score   support

          3       0.00      0.00      0.00         2
          4       0.90      0.95      0.92        19

avg / total       0.81      0.86      0.84        21
'''

print (accuracy_score(df5_test_target, knn_pred_test)) # 0.857142857143
print(mean_squared_error(df5_test_target, knn_pred_test)) # 0.142857142857

# First let's create a dataset called X
X = np.array(dfReviews)
# create train and test
tpercent = 0.85
tsize = int(np.floor(tpercent * len(dfReviews)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
len_train = len(dfReviews_train)
len_test = len(dfReviews_test)
# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.
neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)

for i in range(len_test):
    res = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = res[1]

    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    #print ("Based on product reviews, for ", df_reviews_final["asin"][len_train + i] ," average rating is ",df_reviews_final["overall"][len_train + i])
    #print ("The first similar product is ", df_reviews_final["asin"][first_related_product] ," average rating is ",df_reviews_final["overall"][first_related_product])
    #print ("The second similar product is ", df_reviews_final["asin"][second_related_product] ," average rating is ",df_reviews_final["overall"][second_related_product])
    #print ("-----------------------------------------------------------")

df5_train_target = df_reviews_final["overall"][:len_train]
df5_test_target = df_reviews_final["overall"][len_train:len_train+len_test]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

k_neighbors = 5
knn_clf = neighbors.KNeighborsClassifier(k_neighbors, weights='distance')
knn_clf.fit(dfReviews_train, df5_train_target)
knn_pred_test = knn_clf.predict(dfReviews_test)
#print (knn_pred_test)

print(classification_report(df5_test_target, knn_pred_test))
'''
             precision    recall  f1-score   support

          3       0.50      0.33      0.40         3
          4       0.93      0.97      0.95        29

avg / total       0.89      0.91      0.90        32

'''
print (accuracy_score(df5_test_target, knn_pred_test)) # 0.90625

print(mean_squared_error(df5_test_target, knn_pred_test)) # 0.09375

