# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:39:35 2019

@author: basit
"""

from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.preprocessing import MinMaxScaler

y_tr = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-train-labels.csv',delimiter=',')
y_test = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-test-labels.csv',delimiter=',')
y_test = (y_test < 190)
y_tr = (y_tr < 190)

x_tr = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-train-features.csv',delimiter=',')
x_test = np.genfromtxt('C:/3rd Year/Sixth Semester/CS 464/HW2/question-2-test-features.csv',delimiter=',')
x_test = np.hstack((np.ones((len(x_test),1)),x_test))
x_tr = np.hstack((np.ones((len(x_tr),1)),x_tr))

scaling = MinMaxScaler(feature_range=(-1,1)).fit(x_tr)
x_tr = scaling.transform(x_tr)
x_test = scaling.transform(x_test)

def confusion_measures(x_test,y_test,clf):
    y_pr = clf.predict(x_test).reshape(len(y_test),1)
    y_test = y_test.reshape(len(y_test),1)
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

def split_data(x,k):
    step = int((x_tr.shape[0]/k))
    if len(x.shape) == 1:
        x = x.reshape(len(x),1)
        x_new = np.zeros((k,step,1))
    else:
        x_new = np.zeros((k,step,x.shape[1]))
    for z in range(k):
        if z == k-1:
            x_new[z,:,:] = x[z*step:(z+1)*step,:]
            break
        x_new[z,:,:]= x[z*step:(z+1)*step,:]
    return x_new

def divide(x_folds,test_fold_k,k):
    arr = np.arange(k)
    arr = arr[arr!=test_fold_k]
    data_test = x_folds[test_fold_k,:,:]
    data_train = x_folds[arr,:,:]
    return data_train.reshape((x_folds.shape[0]-1)*x_folds.shape[1],x_folds.shape[2]),data_test


arr = np.arange(len(y_tr))
np.random.shuffle(arr)
x_tr = x_tr[arr]
y_tr = y_tr[arr]

x_folds = split_data(x_tr,10)
y_folds = split_data(y_tr,10)

cost = [0.01, 0.1, 1, 10, 100]
models = []
acc_val = np.zeros((5))
for c in range(5):
    clfs = []
    accs = np.zeros((10))
    for m in range(10):
        clf = SVC(kernel='linear',C= cost[c])
        x_train,x_val = divide(x_folds,m,10)
        y_train,y_val = divide(y_folds,m,10)
        
        clf.fit(x_train, y_train)
        clfs.append(clf)
        confusion,measures = confusion_measures(x_val,y_val,clf)
        acc = measures.get('Accuracy')
        accs[m] = acc
    index_a = np.argmax(accs)
    models.append(clfs[index_a])
    acc_val[c] = np.mean(accs)
index = np.argmax(acc_val)
model = models[index]
confusion,measures_final = confusion_measures(x_test,y_test,model)

##Run until this part only for linear. THen run the next portion for rbf kernel

gamma = [1/16, 1/8, 1/4, 1/2, 1, 2]
models = []
acc_val = np.zeros((6))
for c in range(6):
    clfs = []
    accs = np.zeros((10))
    for m in range(10):
        clf = SVC(kernel='rbf',C= 1,gamma = gamma[c])
        x_train,x_val = divide(x_folds,m,10)
        y_train,y_val = divide(y_folds,m,10)
        clf.fit(x_train, y_train)
        clfs.append(clf)
        confusion,measures = confusion_measures(x_val,y_val,clf)
        acc = measures.get('Accuracy')
        accs[m] = acc
    index_a = np.argmax(accs)
    models.append(clfs[index_a])
    acc_val[c] = 100*np.mean(accs)
index = np.argmax(acc_val)
model = models[index]
confusion,measures_final = confusion_measures(x_test,y_test,model)
