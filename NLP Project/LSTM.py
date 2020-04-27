# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:23:56 2019

@author: basit
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from keras import optimizers
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout,SpatialDropout1D, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

df = pd.read_csv('C:/3rd Year/Sixth Semester/CS 464/Project/text_rating_best_1000.csv', low_memory=False, header = None)
reviews = df[0]
labels = np.array(df[1])
labels = labels - 1

a= []
for review in reviews:
    a.append(len(review))
a = np.array(a)
avg = np.mean(a).astype(int)

MAX_LENGTH = avg
MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = MAX_LENGTH
MAX_NB_WORDS = 1000
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#scaler = MinMaxScaler(feature_range = (-10,10))
#scaler.fit(data)
#data = scaler.transform(data)
#data = data.reshape(2492,100,1)

labels1 = to_categorical(labels)
#labels1 = labels1.reshape(22572,5)

x_train, x_test, y_train, y_test = train_test_split(data, labels1, test_size=0.2)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()

model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=data.shape[1]))

model.add(SpatialDropout1D(0.2))
#,input_shape=(100,1),return_sequences=False
model.add(LSTM(64,dropout=0.2,recurrent_dropout=0.2))

model.add(Dense(5, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
    
print('Train...')

history = model.fit(x_train, y_train, epochs=15,batch_size=32,validation_split=0.1)
plt.plot(history.history['loss'])
plt.show
#model.summary()

x = model.evaluate(x_test, y_test)
z = model.predict(x_train)

print("Loss: ", x[0])
print("Accuracy: ", x[1])
print("Precision: ", x[2])
print("Recall: ", x[3])