# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random as rd
import math

#y_tr = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-train-labels.csv',header = None)
#y_test = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-test-labels.csv',header = None)
#
#x_tr = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-train-features.csv',delimiter=',')
#x_test = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-test-features.csv',delimiter=',')

vocab = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-vocab.txt', delimiter="\t", header = None)
words = vocab[:][0]
occ_words = np.array(vocab[:][1])

indices_pos = y_tr.index[y_tr[0] == 'positive'].tolist()
indices_neg = y_tr.index[y_tr[0] == 'negative'].tolist()
indices_neu = y_tr.index[y_tr[0] == 'neutral'].tolist()

neutral = x_tr[indices_neu,:]
positive = x_tr[indices_pos,:]
negative = x_tr[indices_neg,:]

occ_neutral = np.sum(neutral, axis=0)
occ_positive = np.sum(positive, axis=0)
occ_negative = np.sum(negative, axis=0)

index_neutral = (np.flip(np.argsort(occ_neutral),axis = None))[0:20]
index_positive = (np.flip(np.argsort(occ_positive),axis = None))[0:20]
index_negative = (np.flip(np.argsort(occ_negative),axis = None))[0:20]

words_neu = words[index_neutral]
words_pos = words[index_positive]
words_neg = words[index_negative]
print("Neutral Words:")
print(words_neu)
print("Positive Words:")
print(words_pos)
print("Negative Words:")
print(words_neg)