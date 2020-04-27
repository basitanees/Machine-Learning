# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:41:52 2019

@author: basit
"""
import pandas as pd
#from pandas import DataFrame
import numpy as np

def confusion_measures(x_test,y_test,model):
    y_pr = sigmoid(x_test@model)
    y_pr = y_pr > 0.5
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
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

def logistic_regression(x_train,x_test,y_train,y_test):
    grad = np.ones((x_train.shape[1],1))
    w = np.zeros((x_train.shape[1],1))
    w_pr = 10*np.ones((x_train.shape[1],1))
    n = 0
    L = 0.01
    converge = False
    m = (x_train.shape[1])
    while converge == False:
        hyp = sigmoid(x_train@w)
        grad = (1/m)*(x_train.T@(hyp-y_train))
        w_pr = w
        w = w_pr - (L*grad)
        n = n + 1
    #    if all(abs(w - w_pr) < 0.001):
    #        converge = True
        if all(abs(grad) < 0.01):
            converge = True
    y2 = sigmoid(x_train@w)
    train_cost = (-1/m)*np.sum((y_train*np.log(y2)) + ((1 - y_train) * np.log(1 - y2)) )
    
    trues_tr = (y2>0.5) == y_train
    acc_tr = np.sum(trues_tr)/len(y_train)
    
    y3 = sigmoid(x_test@w)
    test_cost = (-1/m)*np.sum((y_test*np.log(y3)) + ((1 - y_test) * np.log(1 - y3)) )
    trues_test = (y3>0.5) == y_test
    acc_test = np.sum(trues_test)/len(y_test)
    return w,100*acc_test

features_all = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp_academic_dataset_features_all_2.json')
#abc = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp_academic_dataset_review_20.json')
abc = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_user_reviews.json')
users = (abc['user_id']).drop_duplicates()
users = users.reset_index( drop=True)
arr = np.arange(len(users))
np.random.shuffle(arr)
users_new = users.loc[arr[:100]]

num = 0
accur_val = []
accur_tr = []
accur_test = []
measures1 = []
for user_i in users_new:
    feat_u = abc[abc['user_id'] == user_i]
    feat_u['stars'] = 1.0*((feat_u['stars'].values)>3)
    feat_user = feat_u.set_index('business_id').join(features_all.set_index('business_id'))
    feat_user=feat_user.drop(['review_id','user_id'],axis = 1)
    
    cr = np.random.rand(feat_user.shape[0]) < 0.8
    x_train = feat_user[cr].values
    x_test = feat_user[~cr].values
    
    y_test = np.zeros((x_test.shape[0],1))
    y_test[:,0] = x_test[:,0]
    y_train = np.zeros((x_train.shape[0],1))
    y_train[:,0] = x_train[:,0]
    
    tr_len = len(y_train)
    split = tr_len//3
    
    y_train1 = y_train[:split,:]
    y_train2 = y_train[split:2*split,:]
    y_train3 = y_train[2*split:,:]
    
    x_test[:,0] = np.ones((len(y_test)))
    x_train[:,0] = np.ones((len(y_train)))
    x_train1 = x_train[:split,:]
    x_train2 = x_train[split:2*split,:]
    x_train3 = x_train[2*split:,:]
    
    #feat_user.columns.values[74] = 75
    k1 = np.vstack((x_train1,x_train2))
    k2 = np.vstack((x_train1,x_train3))
    k3 = np.vstack((x_train3,x_train2))
    y_k1=np.vstack((y_train1,y_train2))
    y_k2=np.vstack((y_train1,y_train3))
    y_k3=np.vstack((y_train3,y_train2))
    w_1, acc_test_1 = logistic_regression(k1,x_train3,y_k1,y_train3)
    w_2, acc_test_2 = logistic_regression(k2,x_train2,y_k2,y_train2)
    w_3, acc_test_3 = logistic_regression(k3,x_train1,y_k3,y_train1)
    
    w = np.hstack((w_1,w_2,w_3))
    accs = np.hstack((acc_test_1,acc_test_2,acc_test_3))
    acc_val = np.sum(accs)/3  #avg validation accuracy
    model = np.zeros((len(w_1),1))
    model[:,0] = w[:,np.argmax(accs)]
    
    ##train acc on best model
    confusion_tr,measures_tr = confusion_measures(x_train,y_train,model)
    
    #test acc
    confusion_test,measures_test = confusion_measures(x_test,y_test,model)
    measures1.append(measures_test)
    #nans = np.isnan(acr_test)
#    if ~np.isnan(acc_val):
    accur_val.append(acc_val)
    accur_tr.append(measures_tr.get('Accuracy'))
#    if ~np.isnan(measures_tr.get('Precision')):
#        accur_tr.append(measures_tr.get('Precision'))
#    if ~np.isnan(measures_test.get('Recall')):
#        accur_test.append(measures_test.get('Recall'))
    num = num + 1
    print(num)
    
plt.plot(np.arange(len(accur_val)),accur_val,label='Validation')
plt.plot(np.arange(len(accur_tr)),accur_tr,label='Training')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.title('Accuracies of different users')
pylab.legend(loc='upper left')
plt.show()

acc_all = 0
pre_all = 0
rec_all = 0
v = 0
w = 0
for i in range(len(measures1)):
    acc_all = acc_all + measures1[i].get('Accuracy')
    if ~np.isnan(measures1[i].get('Precision')):
        pre_all = pre_all + measures1[i].get('Precision')
        v = v + 1
    if ~np.isnan(measures1[i].get('Recall')):
        rec_all = rec_all + measures1[i].get('Recall')
        w = w + 1
acc_all /= 100
pre_all /= 100-v
rec_all /= 100

accur_tr_avg = np.sum(np.array(accur_tr))/len(accur_tr)
accur_test_avg = np.sum(np.array(accur_test))/len(accur_test)
accur_val_avg = np.sum(np.array(accur_val))/len(accur_val)