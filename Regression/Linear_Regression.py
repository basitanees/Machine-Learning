# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 01:31:36 2019

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

def linear_regression(x_tr,y_tr):
    x_x = x_tr.T@x_tr
#    rank = np.linalg.matrix_rank(x_x)
    B = np.linalg.inv(x_x)@x_tr.T@y_tr    
    return B

B = linear_regression(x_tr,y_tr)
J_tr = (y_tr-x_tr@B).T@(y_tr-x_tr@B)/(len(y_tr))
J_test = (y_test-x_test@B).T@(y_test-x_test@B)/len(y_test)

hum = np.hstack((x_tr[:,7],x_test[:,7]))
hum = hum.reshape(len(hum),1)
bikes = np.hstack((y_tr,y_test))
#bikes_pr_1 = np.hstack((x_tr@B,x_test@B))
bikes_pr_1 = hum*B[7]

B_new = linear_regression(x_tr[:,7].reshape((len(x_tr),1)),y_tr)

bikes_pr_2 = hum*B_new
B=B_new
#J_tr = (y_tr-x_tr[:,7]*B).T@(y_tr-x_tr[:,7]*B)/(len(y_tr))
#J_test = (y_test-x_test[:,7]*B).T@(y_test-x_test[:,7]*B)/len(y_test)

colors = (0,0,0)
area = np.pi*3
plt.scatter(hum,bikes_pr_1,s = area, c = colors,alpha = 0.5)
plt.title('Bike count vs normalized humidity-Trained on all features')
plt.xlabel('normalized humidity')
plt.ylabel('Bike count')
plt.show()

plt.scatter(hum,bikes_pr_2,s = area, c = colors,alpha = 0.5)
plt.title('Bike count vs normalized humidity-Trained on humidity feature')
plt.xlabel('normalized humidity')
plt.ylabel('Bike count')
plt.show()

