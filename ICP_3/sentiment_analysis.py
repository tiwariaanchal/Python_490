from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from datetime import time
import tensorflow as tf




df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train =fetch_20newsgroups(subset='train', shuffle=True)
# sentences = newsgroups_train.data
# y = newsgroups_train.target
#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
# sentences = tokenizer.texts_to_matrix(sentences)

max_review_len= max([len(s.split()) for s in sentences])
vocab_size= len(tokenizer.word_index)+1
sentences = tokenizer.texts_to_sequences(sentences)
padded_docs= pad_sequences(sentences,maxlen=max_review_len)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)

model = Sequential()

model.add(Embedding(vocab_size, 100, input_length=max_review_len))
model.add(Flatten())
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])

# validation_data=(arr_x_valid, arr_y_valid),callbacks=[keras.callbacks.TensorBoard(log_dir="logs/final", histogram_freq=1, write_graph=True, write_images=True)])

tensorboard = TensorBoard(log_dir="logs/final", histogram_freq=1, write_graph=True, write_images=False)

history = model.fit(X_train,y_train,verbose=1,validation_data=(X_test,y_test),callbacks=[tensorboard])

# history=model.fit(X_train,y_train, epochs=1, verbose=True, validation_data=(X_test,y_test), batch_size=256, callbacks=[tensorboard])

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# predict_test = model.predict_classes(X_test[[30], :])
# print("The prediction of the 30th in the test dataset is: ", predict_test)
#
# plt.imshow(test_images[30,:,:],cmap='gray')
# plt.show()