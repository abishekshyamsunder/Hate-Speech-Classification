
# Commented code for creating the embedding matrix from the stanford corpus
# embeddings_index = dict()
# f = open('../input/glove6b/glove.6B.300d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# embedding_matrix = np.zeros((vocab_size, 300))
# for word, i in tokenizer.word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
# with open('embedding_matrix.npy') as f:
#     np.save(embedding_matrix, x)

from gc import callbacks
from function_generators import *
import nltk
import warnings
warnings.filterwarnings("ignore")
nltk.download('stopwords')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pandas as pd

from termcolor import colored

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

tf.random.set_seed(1234)
np.random.seed(1234)

import string
import time
table = str.maketrans('', '', string.punctuation)

from sklearn.preprocessing import FunctionTransformer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

import csv

combined_df = pd.read_csv("data/clean/combined.csv", index_col = "Unnamed: 0")
num_to_sample = np.sum(combined_df['Label']==1)
df_zero = combined_df.query("Label==0").sample(n = num_to_sample, random_state=1)
df_one = combined_df.query("Label==1")
combined_df = df_zero.append(df_one, ignore_index=True)
combined_df = combined_df.sample(frac = 1)
vocab_size = 17667

reddit_df = pd.read_csv("data/clean/reddit.csv", index_col = "Unnamed: 0")
num_to_sample = np.sum(reddit_df['Label']==1)
df_zero = reddit_df.query("Label==0").sample(n = num_to_sample, random_state=1)
df_one = reddit_df.query("Label==1")
reddit_df = df_zero.append(df_one, ignore_index=True)
reddit_df = reddit_df.sample(frac = 1)
vocab_size = 17667


cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training_1/cp.ckpt",save_weights_only=True,verbose=1)

with open("embedding_mat.npy","rb") as f:
    embedding_matrix = np.load(f)


def create_transfer_model(optimizer='adagrad', kernel_initializer='glorot_uniform', dropout=0.2):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Embedding(vocab_size,300,weights = [embedding_matrix],input_length=300,trainable = True),
      tf.keras.layers.GRU(units = 6, dropout=0.3, activation="tanh"),
      tf.keras.layers.Dense(units = 10,activation="relu"),
      tf.keras.layers.Dense(units = 1, activation="sigmoid")
  ])
  model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy','binary_crossentropy'])
  return model

model1 = KerasClassifier(build_fn=create_transfer_model,epochs = 10,verbose=1)


models = [model1]
names = ["Glove Model"]

for i in range(len(models)):
  pipe = Pipeline([('Convert to lower', ConvertToLower('Data')),
                  ('Remove Punctuation', RemovePunctuation('Data')),
                  ('Remove Stop Words', RemoveStopWords('Data')),
                  ('Stem Input', StemInput('Data')),
                  ('Tokenise Input', TokeniseInput('Data')),
                  ('Pad Sequences', PadSequences('sequences', maxlen=300)),
                  ('Model', models[i])
                  ])

  start_train = time.time()
  (train, test) = train_test_split(combined_df, test_size=0.2, random_state=42, shuffle=True)
  pipe.fit(train[['Data']], train['Label'])
  end_train = time.time()
  start_test = time.time()
  accuracy = pipe.score(test[['Data']], test['Label'])
  end_test = time.time()
  pred_labels = pipe.predict(test[['Data']])
  f1_sc = f1_score(test[['Label']], pred_labels, average='macro')
  train_time = end_train - start_train
  test_time = end_test - start_test
  print(colored("The {name} model gives us an accuracy of: {accuracy} for combined".format(name = names[i], accuracy = accuracy), 'green'))
  print(colored("The {name} model has a training time of: {train_time} for combined".format(name = names[i], train_time = train_time), 'green'))
  print(colored("The {name} model has a testing time of: {test_time} for combined".format(name = names[i], test_time = test_time), 'green'))
  print(colored("The {name} model has a f1_score of: {f1_score} for combined".format(name = names[i], f1_score = f1_sc), 'green'))



  with open('stats.csv', 'a', newline='') as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(['{name}'.format(name = names[i]), 'Combined',accuracy, f1_sc, train_time, test_time/len(test[['Label']])])

  start_train = time.time()
  (train_reddit, test_reddit) = train_test_split(reddit_df, test_size=0.2, random_state=42, shuffle=True)
  pipe.fit(train_reddit[['Data']], train_reddit['Label'])
  end_train = time.time()
  start_test = time.time()
  accuracy = pipe.score(test_reddit[['Data']], test_reddit['Label'])
  end_test = time.time()
  pred_labels = pipe.predict(test_reddit[['Data']])
  f1_sc = f1_score(test_reddit[['Label']], pred_labels, average='macro')
  train_time = end_train - start_train
  test_time = end_test - start_test
  print(colored("The {name} model gives us an accuracy of: {accuracy} for Reddit".format(name = names[i], accuracy = accuracy), 'yellow'))
  print(colored("The {name} model has a training time of: {train_time} for Reddit".format(name = names[i], train_time = train_time), 'yellow'))
  print(colored("The {name} model has a testing time of: {test_time} for Reddit".format(name = names[i], test_time = test_time), 'yellow'))
  print(colored("The {name} model has a f1_score of: {f1_score} for Reddit".format(name = names[i], f1_score = f1_sc), 'yellow'))

  with open('stats.csv', 'a', newline='') as csvfile:
      spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
      spamwriter.writerow(['{name}'.format(name = names[i]), 'Reddit',accuracy, f1_sc, train_time, test_time/len(test[['Label']])])
