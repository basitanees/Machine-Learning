# -*- coding: utf-8 -*-
"""
Created on Sun May 12 01:18:24 2019

@author: basit
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 11 21:47:24 2019

@author: basit
"""
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout,SpatialDropout1D, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

#df = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/Project/1429_1.csv', low_memory=False, usecols=['reviews.text', 'reviews.rating'])
#df = df.rename(columns={'reviews.rating' : 'Rating', 'reviews.text' : 'Text'})
#df=df.dropna()
#df[['Rating', 'Text']].applymap(lambda x: str(x).encode('utf-8').decode('ascii', 'ignore'))
#df['Text'] = df['Text'].str.replace('[^\w\s]','')
df = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/Project/amazon_ratings_2_balanced.csv')
df = df.sample(frac=1).reset_index(drop=True)
reviews = df['Text']
labels = df['Rating'].values - 1

labels = labels > 3

#plt.hist(df.Rating.values, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
#plt.title('Amazon')
#plt.ylabel('Frequency')
#plt.xlabel('Ratings')
#
#plt.show()

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
STOPWORDS = set(stopwords.words('english'))

review_clean = []
for text in reviews:
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    review_clean.append(text)

a= []
for review in review_clean:
    a.append(len(review.split()))
a = np.array(a)
avg_a = (np.mean(a)).astype(int)

MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 32

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(review_clean)
sequences = tokenizer.texts_to_sequences(review_clean)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

b= []
for review in sequences:
    b.append(len(review))
b = np.array(b)
avg_b = (np.mean(b)).astype(int)

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels1 = to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels1, test_size=0.2)

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#x_train = x_train.reshape(40000,100,1)
#y_train= y_train.reshape(40000,5)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=data.shape[1]))
model.add(SpatialDropout1D(0.2))
#,input_shape=(100,1),return_sequences=False
model.add(LSTM(64,input_shape=(100, 1)))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
    
print('Train...')

history = model.fit(x_train, y_train, epochs=10,batch_size=64,validation_split=0.1)
#,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
model_hist = history.history
plt.plot(history.history['loss'])
plt.show
#model.summary()
model.save('C:/3rd Year/Sixth Semester/CS 464/Project/my_model_10000.h5')

m2 = np.argmax(y_test,axis = 1).reshape(y_test.shape[0],1)

z = model.predict(x_test)
m = np.argmax(z,axis = 1).reshape(z.shape[0],1)

acc_test = np.sum(m == m2)/len(y_test)
#m2 = np.argmax(y_test,axis = 1)
#m2 = y_test
acc = []
for i in range(2):
    cr = m2 == i
    m21 = m2[cr]
    m1 = m[cr]
    acc.append(np.sum(m1 == m21)/len(m21))
print("Loss: ", x[0])

plt.hist(m2+1, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
plt.title('Amazon')
plt.ylabel('Frequency')
plt.xlabel('Ratings')

plt.show()