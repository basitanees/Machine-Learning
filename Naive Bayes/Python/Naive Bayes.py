# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random as rd
import math

#use the respective directory while running the code in a different computer
#y_tr = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-train-labels.csv',header = None)
#y_test = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-test-labels.csv',header = None)
#
#x_tr = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-train-features.csv',delimiter=',')
#x_test = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW1/question-4-test-features.csv',delimiter=',')

indices_pos = y_tr.index[y_tr[0] == 'positive'].tolist()
indices_neg = y_tr.index[y_tr[0] == 'negative'].tolist()
indices_neu = y_tr.index[y_tr[0] == 'neutral'].tolist()

neutral = x_tr[indices_neu,:]
positive = x_tr[indices_pos,:]
negative = x_tr[indices_neg,:]

def mle(tweets, alpha):
    occ_words = np.sum(tweets, axis=0) + alpha
    total_words = np.sum(occ_words) 
    theta = occ_words/total_words
    return theta

#use alpha = 1 for applying smoothing
alphaa = 1
theta_neutral = mle(neutral,alphaa)
theta_positive = mle(positive,alphaa)
theta_negative = mle(negative,alphaa)

pr_pos = len(indices_pos)/11712
pr_neg = len(indices_neg)/11712
pr_neu = len(indices_neu)/11712

priors = np.array([pr_neu,pr_pos,pr_neg])
likelihoods = np.array([theta_neutral,theta_positive,theta_negative])
#labels = ['neutral', 'positive','negative']
labels_pr = []
#posteriors = np.zeros((2928,3))

prob = np.zeros((3, 5722))

inter = (x_test * np.log(likelihoods[0,:]))
inter[np.isnan(inter)] = 0
prob1 = np.log(priors[0]) + np.sum(inter, axis = 1)

inter = (x_test * np.log(likelihoods[1,:]))
inter[np.isnan(inter)] = 0
prob2 = np.log(priors[1]) + np.sum(inter, axis = 1)

inter = (x_test * np.log(likelihoods[2,:]))
inter[np.isnan(inter)] = 0
prob3 = np.log(priors[2]) + np.sum(inter, axis = 1)

probs = np.vstack((prob1,prob2,prob3)).T
labels_out = np.argmax(probs, axis = 1)
for i in range(len(labels_out)):
    labels_pr.append(labels[labels_out[i]])
labels_pr = pd.DataFrame(labels_pr)

trues = labels_pr == y_test
accuracy = (np.sum(trues)/2928)*100
if alphaa == 0:
    print("The accuracy using multinomial is: ")
    print(accuracy)
else:
    print("The accuracy using multinomial with smoothing is: ")
    print(accuracy)