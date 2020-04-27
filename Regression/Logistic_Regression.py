# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 02:59:07 2019

@author: basit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

y_tr = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-train-labels.csv',delimiter=',')
y_test = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-test-labels.csv',delimiter=',')

x_tr = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-train-features.csv',delimiter=',')
x_test = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-test-features.csv',delimiter=',')
x_test = np.hstack((np.ones((len(x_test),1)),x_test))
x_tr = np.hstack((np.ones((len(x_tr),1)),x_tr))

y_test = (y_test < 190)
y_tr = (y_tr < 190)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression_batch(x_train,y_train,L):
    w = np.zeros((x_train.shape[1],1))
    n = 0
    m = (x_train.shape[0])
    while n < 1000:
        hyp = sigmoid(x_train@w)
        grad = (1/m)*(x_train.T@(y_train.reshape(m,1)-hyp))
        w = w + (L*grad)
        n = n + 1
#        y2 = sigmoid(x_train@w)
#        train_cost = (-1/m)*np.sum((y_train*np.log(y2)) + ((1 - y_train) * np.log(1 - y2)) )
#        trues_tr = (y2>0.5) == y_train
#        acc_tr = np.sum(trues_tr)/len(y_train)
    return w

def logistic_regression_stochastic(x_train,y_train,L):
    w = np.zeros((x_train.shape[1],1))
    n = 0
    m = (x_train.shape[1])
#    np.random.shuffle(x_train)
    arr = np.arange(len(y_train))
    np.random.shuffle(arr)
    x_train = x_train[arr]
    y_train = y_train[arr]
    for i in range(1000):
        while n < 1000:
            hyp = sigmoid(x_train[n,:]@w)
            grad = (x_train[n,:].T*(y_train[n]-hyp)).reshape(m,1)
            w = w + (L*grad)
            n = n + 1
        return w

def logistic_regression_mini_batch(x_train,y_train,L,batch_size):
    w = np.zeros((x_train.shape[1],1))
    bs = batch_size
    n = 0
#    np.random.shuffle(x_train)
    arr = np.arange(len(y_train))
    np.random.shuffle(arr)
    x_train = x_train[arr]
    y_train = y_train[arr]
    for i in range(1000):
        while n < 437:
            hyp = sigmoid(x_train[n*bs:bs*(n+1),:]@w)
            grad = (1/bs)*(x_train[n*bs:bs*(n+1),:].T@(y_train[n*bs:bs*(n+1)].reshape(bs,1)-hyp))
            w = w + (L*grad)
            n = n + 1
        return w

def confusion_matrix(B,x_test,y_test):
    y_pr = sigmoid(x_test@B) > 0.5
    y_test = y_test.reshape(len(y_test),1)
#    test_cost = (-1/m)*np.sum((y_test*np.log(y)) + ((1 - y_test) * np.log(1 - y)) )
#    trues_test = y_pr == y_test
#    acc_test = 100*(np.sum(trues_test)/len(y_test))
    tp = np.sum(np.logical_and(y_pr, y_test))
    tn = np.sum(np.logical_and(np.logical_not(y_pr), np.logical_not(y_test)))
    fp = np.sum(np.logical_and(y_pr, np.logical_not(y_test)))
    fn = np.sum(np.logical_and(np.logical_not(y_pr), y_test))
    confusion = np.array(([tp,fp],[fn,tn]))
    acc_test = 100*(tp+tn)/(tp+tn+fp+fn)
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    npv = tn/(tn+fn)
    fpr = fp/(fp+tn)
    fdr = fp/(tp+fp)
    f1 = 2*pre*rec/(pre+rec)
    b = 2
    f2 = ((1+(b*b))*pre*rec)/((b*b*pre)+rec)
    measures = {'Accuracy': acc_test,'Precision': pre,'Recall': rec,'NPV': npv,'FPR': fpr,'FDR': fdr,'F1': f1,'F2': f2}
    return confusion, measures


L = 0.1
B1 = logistic_regression_batch(x_tr,y_tr,L)
confusion1,measures1 = confusion_matrix(B1,x_test,y_test)

B2 = logistic_regression_mini_batch(x_tr,y_tr,L,32)
confusion2,measures2 = confusion_matrix(B2,x_test,y_test)

B3 = logistic_regression_stochastic(x_tr,y_tr,L)
confusion3,measures3 = confusion_matrix(B3,x_test,y_test)
