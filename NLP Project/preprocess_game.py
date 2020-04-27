# -*- coding: utf-8 -*-
"""
Created on Sun May 12 00:57:33 2019

@author: basit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce

#df = pd.read_json('C:/3rd Year/Sixth Semester/CS 464/Project/Video_Games_5.json',lines=True)
#df = df[['overall','reviewText']]
#df = df.rename(columns={'overall' : 'Rating', 'reviewText' : 'Text'})
#df=df.dropna()
#df[['Rating', 'Text']].applymap(lambda x: str(x).encode('utf-8').decode('ascii', 'ignore'))
#df['Text'] = df['Text'].str.replace('[^\w\s]','')
#df = df[df.Text != '']
#df.to_csv('C:/3rd Year/Sixth Semester/CS 464/Project/amazon_ratings_2.csv', index=False)
df = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/Project/amazon_ratings_2.csv')

plt.hist(df.Rating.values, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.title('Amazon')
plt.ylabel('Frequency')
plt.xlabel('Ratings')

plt.show()

a = df.groupby('Rating').count()
b1 = df[df.Rating == 1.0]
b2 = df[df.Rating == 2.0]
b3 = df[df.Rating == 3.0]
b4 = df[df.Rating == 4.0]
b5 = df[df.Rating == 5.0]

b11 = b1[0:10000]
b12 = b2[0:10000]
b13 = b3[0:10000]
b14 = b4[0:10000]
b15 = b5[0:10000]

dfs = pd.concat([b11, b12, b13, b14, b15])

dfs.to_csv('C:/3rd Year/Sixth Semester/CS 464/Project/amazon_ratings_2_balanced.csv', index=False)

plt.hist(dfs.Rating.values, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.title('Amazon')
plt.ylabel('Frequency')
plt.xlabel('Ratings')

plt.show()
