# -*- coding: utf-8 -*-
"""
Created on Wed May 15 05:19:26 2019

@author: basit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab

def ReLu(z):
    return z*(z > 0)

def ReLuGradient(z):
    return z > 0

#def Softmax(Z):
#    total = np.sum(np.exp(Z),axis = 0)
#    return np.exp(Z)/total
    
def Softmax(x):
    e_x = np.exp(x - np.max(x,axis = 1).reshape(x.shape[0],1))
    return e_x / np.sum(e_x,axis = 1).reshape(x.shape[0],1)

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

def to_categorical(y):
    out = np.zeros((len(y),2))
    for i in range(2):
        out[:,i] = (y == i).reshape(len(y))
    return out

def val_split(x_tr, y_tr,percent):
    dat = np.hstack((y_tr,x_tr[:,1:]))
    return train_test_split(dat, percent)
    
def forward(x_train):
    global z2,a2,z3,a3,z4,theta_l1,theta_l2,theta_out
    z2 = x_train@theta_l1.T
    a2 = np.hstack((np.ones((z2.shape[0],1)),ReLu(z2)))
    z3 = a2 @ theta_l2.T
    a3 = np.hstack((np.ones((z3.shape[0],1)),ReLu(z3)))
    z4 = a3 @ theta_out.T
    h2 = Softmax(z4)
    return h2
def back(x,y , h1):
    global theta1_grad,theta2_grad,theta3_grad,delta4,delta3,delta2,z2,a2,z3,a3,z4,h,lem
    delta4 = (h1 - y)
    
    delta3 = (delta4@theta_out)*(ReLuGradient(np.hstack((np.ones((z3.shape[0],1)),z3))))
    delta3 = delta3[:,1:]
    
    delta2 = (delta3@theta_l2) * (ReLuGradient(np.hstack((np.ones((z2.shape[0],1)),z2))))
    delta2 = delta2[:,1:]
    
    theta3_grad = delta4.T @ a3
    theta2_grad = delta3.T @ a2
    theta1_grad = delta2.T @ x

def accuracy(h,y):
    h_tr = np.argmax(h,axis = 1).reshape(h.shape[0],1)
    acc_tr = np.sum(y == h_tr)/len(y)
    return acc_tr

def cost(h,y):
    cost = np.sum( (-y*np.log10(h)) - ((1-y)*np.log10(1-h)) )/len(y)
    return cost

def confusion_measures(h_test,y_test):
    y_pr = np.argmax(h_test,axis = 1).reshape(h_test.shape[0],1)
#    y_pr = y_pr > 0.5
    y_test = y_test.reshape(len(y_test),1)
    tp = np.sum(np.logical_and(y_pr, y_test))
    tn = np.sum(np.logical_and(np.logical_not(y_pr), np.logical_not(y_test)))
    fp = np.sum(np.logical_and(y_pr, np.logical_not(y_test)))
    fn = np.sum(np.logical_and(np.logical_not(y_pr), y_test))
    confusion = np.array(([tp,fn],[fp,tn]))
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

features_all = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp_academic_dataset_features_all_2.json')
abc = pd.read_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp-dataset/yelp_academic_dataset_user_reviews.json')
users = (abc['user_id']).drop_duplicates()
users = users.reset_index( drop=True)
arr = np.arange(len(users))
np.random.shuffle(arr)
users1 = users.loc[arr[:50]]

measuress = []
plots = []
for person,index in users1:
    user = abc[abc['user_id'] == person]
    user['stars'] = 1.0*((user['stars'].values)>3)
    feat_user = user.set_index('business_id').join(features_all.set_index('business_id'))
    feat_user=feat_user.drop(['review_id','user_id'],axis = 1)
    data = feat_user.values
    n = abc.groupby('user_id').count()
    x_test, x_train, y_test, y_train = train_test_split(data,0.8)
    #p = np.nonzero(y_train == 1)
    #q = x_train[p[0],:]
    #r = y_train[p[0],:]
    #x_train = np.vstack((x_train,q))
    #y_train = np.vstack((y_train,r))
    #p = pd.DataFrame({'label':[y_train],'data':[x_train]}, index=[0])
    
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    
    num_columns = 76
    num_layer1 = 80
    num_layer2 = 50
    num_layer_out = 2
    num_train = x_train.shape[0] 
    L = 0.002
    epochs = 30
    lem = 100
    
    theta_l1 = np.random.rand(num_layer1,num_columns) -0.5
    theta_l2 = np.random.rand(num_layer2,num_layer1+1)-0.5
    theta_out = np.random.rand(num_layer_out,num_layer2+1)-0.5
    
    h = forward(x_train)
    back(x_train,y_train,h)
    #Ji = np.sum( (-y_train*np.log(h)) - ((1-y_train)*np.log(1-h)) )/num_train;
    J_tr = []
    J_val = []
    acc_train = []
    acc_val = []
    y_tr = np.argmax(y_train,axis = 1).reshape(y_train.shape[0],1)
    y_tt = np.argmax(y_test,axis = 1).reshape(y_test.shape[0],1)
    
    x_val, x_tr1, y_val, y_tr1 = val_split(x_train, y_tr,0.8)
    
    #p = np.nonzero(y_tr1 == 1)
    #q = x_tr1[p[0],:]
    #r = y_tr1[p[0],:]
    #x_tr1 = np.vstack((x_tr1,q))
    #y_tr1 = np.vstack((y_tr1,r))
    #arr = np.arange(len(y_tr1))
    #np.random.shuffle(arr)
    #x_tr1 = x_tr1[arr,:]
    #y_tr1 = y_tr1[arr]
    #y_tr1 = y_train = to_categorical(y_tr1)
    #y_val = y_train = to_categorical(y_val)
    #    y_vals = np.argmax(y_val,axis = 1).reshape(y_val.shape[0],1) 
    #unique, index = np.unique(x_val, axis=0, return_index=True)
    #x_val = unique
    #y_val = y_val[index]
    num_train = len(y_tr1)
    for j in range(epochs):   
        for i in range(num_train):
            x = x_tr1[i,:].reshape(1,x_tr1.shape[1])
            y = to_categorical(y_tr1[i,:]).reshape(1,len(h[i,:]))
        #    h2 = h[i,:].reshape(1,len(h[i,:]))
            h2 = forward(x)
            back(x,y, h2)
            theta_l1 = theta_l1 - L*(theta1_grad)
            theta_l2 = theta_l2 - L*(theta2_grad)
            theta_out = theta_out - L*(theta3_grad)
        h_tr1 = forward(x_tr1)
        acc_tr = accuracy(h_tr1,y_tr1)
        Jf = cost(h_tr1,y_tr1);
        J_tr.append(Jf)
        acc_train.append(acc_tr)
        
        h_val = forward(x_val)
        acc_v = accuracy(h_val,y_val)
        Jv = cost(h_val,y_val)
        J_val.append(Jv)
        acc_val.append(acc_v)
        
#        print('epoch: ' , j)
#        print('J_tr: ' ,"%.4f" % Jf, "J_val: ","%.4f" %  Jv, 'acc_tr: ' ,"%.4f" % acc_tr, "acc_val: ","%.4f" %  acc_v)
    print('user: ' , index)    
    h_test = forward(x_test)
    acc_test = accuracy(h_test,y_tt)
    confusion,measures = confusion_measures(h_test, y_tt)
    
    plots.append({'val':acc_val,'train':acc_train})
    measuress.append(measures)
for index in range(20):
    acc_val = plots[index].get('val')
    acc_train = plots[index].get('train')
    plt.plot(np.arange(len(acc_val)),acc_val,label='Validation')
    plt.plot(np.arange(len(acc_train)),acc_train,label='Training')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    pylab.legend(loc='upper left')
    plt.show()

acc_all = 0
pre_all = 0
rec_all = 0
for i in range(len(measuress)):
    acc_all = acc_all + measuress[i].get('Accuracy')
    pre_all = pre_all + measuress[i].get('Precision')
    rec_all = rec_all + measuress[i].get('Recall')
acc_all /= 50
pre_all /= 50
rec_all /= 50