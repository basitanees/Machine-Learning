# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:53:54 2019

@author: basit
"""
import pandas as pd
import numpy as np

business = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_business_new1.json')
review_rating = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_review_rating.json')
#abc = review_rating.groupby('user_id').filter(lambda x: len(x) > 50)
#abc.to_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_user_reviews.json')

#abc = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_user_reviews.json')
abc = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp_academic_dataset_review_20.json')

ratings_user = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_cluster_ratings_predicted.json')
ratings_user = ratings_user.values
# Extract businesses meeting certain criteria
users_count = abc.groupby('business_id').filter(lambda x: len(x) > 300)
businesses = (users_count['business_id']).drop_duplicates()
restaurants = []
for thing in businesses:
    restaurant = users_count[users_count['business_id'] == thing]
    stars = restaurant.groupby('stars').count()
    new = {'business' : restaurant, 'stars': stars, 'total': len(restaurant)}
    restaurants.append(new)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
# Vectorize
vectorizer_categories = CountVectorizer(min_df = 0.01, max_df = 0.8, tokenizer = lambda x: x.split(', '))
vectorized_categories = vectorizer_categories.fit_transform(business['categories'])
vc = vectorized_categories.toarray()
# Apply k means clustering
kmeans_ct = KMeans(n_clusters = 25)
kmeans_ct.fit(vc)
labels_tr = kmeans_ct.predict(vc)
centroids_tr_ct = kmeans_ct.cluster_centers_
print(kmeans_ct.inertia_)

clr = pd.DataFrame(labels_tr,columns = ['cluster'])

business_new = business.reset_index()
business_new = business_new.drop(['index'],axis = 1)
business_new = business_new.join(clr)

business_new.to_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_business_cluster_inc.json')

word = []
count = []
for i in range(25):
    business0 = business_new[business_new['cluster']==i]
    if business0.shape[0] > 0:
        vectorizer_categories = CountVectorizer(min_df = 0.10, max_df = 1.0, tokenizer = lambda x: x.split(', '))
        vectorized_categories = vectorizer_categories.fit_transform(business0['categories'])
        ct = vectorized_categories.toarray()
        cats = vectorizer_categories.get_feature_names()
        ind =  cats.index('restaurants')
        cats.remove('restaurants')
        if 'food' in cats:
            cats.remove('food')
        num1 = np.sum(ct,axis = 0)
        arr = np.arange(len(num1))
        arr=arr[arr!=ind]
        num = num1[arr]
        if len(cats) > 0:
            word.append(cats[np.argmax(num)])
            count.append(num[np.argmax(num)])

# Use matrix factorization to classify clusters       
clusterses = pd.DataFrame({'Category':word,'Count':count})
user = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp_academic_dataset_user_tr.json')
user1 = user[user['review_count']>2000]
userids = user1['user_id'].tolist('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_user_reviews.json')

ratings_user = np.zeros((len(userids),25))
for i in range(len(userids)):
    cr = abc['user_id'] == (userids[i])
    user_rev = abc[cr][['user_id','business_id','stars']]
    cr1 = business_new['business_id'].isin(user_rev['business_id'])
    clusters = business_new[cr1][['business_id','cluster']]
    user_rev3 = user_rev.set_index('business_id').join(clusters.set_index('business_id'))
    
    m = user_rev3.groupby('cluster').mean().stars
    ratings_user[i,m.keys()]= m.values
    
R = ratings_user
#R = numpy.asarray(R)

N = len(R)
M = len(R[0])
K = 2

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R, P, Q, K)
rating_user_pr = nP@nQ.T

ratings_user1 = pd.DataFrame(ratings_user)
ratings_user_pr1 = pd.DataFrame(rating_user_pr)

ratings_user1.to_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_cluster_ratings.json')
ratings_user_pr1.to_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_cluster_ratings_predicted.json')

def matrix_factorization(R, P, Q, K, steps=4000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        if e < 0.001:
            break
    return P, Q.T

def train_test_split(data,train_percent):
    np.random.shuffle(data)
    y = data[:,0].copy().reshape(data.shape[0],1)
    cr = int(train_percent * data.shape[0])
    x_train = data[:cr,:].copy()
    x_test = data[cr:,:].copy()
    y_test = y[cr:]
    y_train = y[:cr]
    x_test[:,0] = np.ones((x_test.shape[0]))
    x_train[:,0] = np.ones((x_train.shape[0]))
    return x_test, x_train, y_test, y_train