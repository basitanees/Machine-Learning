# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:19:51 2019

@author: basit
"""
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
#business_train2

# Data processing
word = []
count = []
for i in range(25):
    business0 = business_train[business_train['cluster']==i]
    vectorizer_categories = CountVectorizer(min_df = 0.40, max_df = 1.0, tokenizer = lambda x: x.split(', '))
    vectorized_categories = vectorizer_categories.fit_transform(business0['categories'])
    vc = vectorized_categories.toarray()
    num = np.sum(vc,axis = 0)
    cats = vectorizer_categories.get_feature_names()
    ind =  cats.index('restaurants')
    cats.remove('restaurants')
    if 'food' in cats:
        cats.remove('food')
    num1 = np.sum(vc,axis = 0)
    arr = np.arange(len(num1))
    arr=arr[arr!=ind]
    num = num1[arr]
    if len(cats) > 0:
        word.append(cats[0])
        count.append(num[0])

#a = np.hstack((rng.normal(size=1000),
#rng.normal(loc=5, scale=2, size=1000)))
#plt.hist(a, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()