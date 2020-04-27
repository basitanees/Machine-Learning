# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 04:59:23 2019

@author: basit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
    
folder = 'C:/3rd Year/Sixth Semester/CS 464/HW2/lfwdataset/lfwdataset/'
images = np.zeros((1000,64*64))
j = 0
for filename in os.listdir(os.getcwd()):
    images[j,:] = (mpimg.imread(folder+filename)).flatten()
    j = j + 1
    if j == 1000:
        break
    
img = mpimg.imread('C:/3rd Year/Sixth Semester/CS 464/HW2/lfwdataset/lfwdataset/Asif_Ali_Zardari_0001.pgm')     
plt.imshow(img, plt.cm.gray)
plt.show()

x = images
x_avg = np.mean(x,axis=0)
x = x - x_avg

cov = x.T@x
e, v = np.linalg.eig(cov)
sorted_index = np.flip(np.argsort(e))
sorted_v = v[:,sorted_index]

z = x@sorted_v
z_sq = z**2
var_total = np.sum(x**2)    

vals = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
pve = []
for k in vals:
    pca_var = np.sum(z_sq[:,:k])
    pv = pca_var/var_total
    pve.append(np.real(pv))

plt.plot(vals,pve)
plt.xlabel('k')
plt.ylabel('PVE')
plt.title('k vs explained variance')
plt.show()

x_new_10_k = (240*sorted_v[:,:10].T+ 16 ).reshape(10,64,64)
x_new_32 = (x[:10,:]@sorted_v[:,:32]@sorted_v[:,:32].T).reshape(10,64,64) +x_avg.reshape(64,64)
x_new_128 = (x[:10,:]@sorted_v[:,:128]@sorted_v[:,:128].T).reshape(10,64,64) +x_avg.reshape(64,64)
x_new_512 = (x[:10,:]@sorted_v[:,:512]@sorted_v[:,:512].T).reshape(10,64,64) + x_avg.reshape(64,64)
x = x.reshape(1000,64,64)[:10,:,:] + x_avg.reshape(64,64)

x_all = np.zeros((5,10,64,64))
x_all[0,:,:,:]=x
x_all[1,:,:,:]=x_new_10_k
x_all[2,:,:,:]=x_new_32
x_all[3,:,:,:]=x_new_128
x_all[4,:,:,:]=x_new_512

a, grid = plt.subplots(nrows=5,ncols=10)

for p in range(5):
    for q in range(10):
        grid[p,q].imshow(np.real(np.round(x_all[p,q,:,:])), plt.cm.gray)
        grid[p,q].axes.get_xaxis().set_visible(False)
        grid[p,q].axes.get_yaxis().set_visible(False)

