from gc import callbacks
from function_generators import *
import nltk
import warnings
import sys
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

import string
import time
table = str.maketrans('', '', string.punctuation)

from sklearn.preprocessing import FunctionTransformer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def predict_scores(path_to_test):
    combined_df = pd.read_csv("data/clean/combined.csv", index_col = "Unnamed: 0")
    num_to_sample = np.sum(combined_df['Label']==1)
    df_zero = combined_df.query("Label==0").sample(n = num_to_sample, random_state=1)
    df_one = combined_df.query("Label==1")
    combined_df = df_zero.append(df_one, ignore_index=True)
    combined_df = combined_df.sample(frac = 1)
    vocab_size = 10000

    test_df = pd.read_csv(path_to_test, index_col = "Unnamed: 0")
    
    def create_dense_batch_norm_model(optimizer='adagrad', kernel_initializer='glorot_uniform', dropout=0.2):
        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Embedding(vocab_size,16,input_length=120), # input embedding learnt of length 16
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(units = 10,activation="relu"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dense(units = 1, activation="sigmoid")
        # ])
        model = tf.keras.models.load_model("training_1/cp.ckpt")
        # model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy','binary_crossentropy'])
        # model.load_weights("training_1/cp.ckpt")
        return model

    model = create_dense_batch_norm_model()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training_1/cp.ckpt",save_weights_only=True,verbose=1)
    test_df['Data'] = test_df['Data'].str.lower()
    test_df['Data'] = test_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in str(x).split()]))
    test_df['Data'] = test_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    test_df['Data'] = test_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))
    
    combined_df['Data'] = combined_df['Data'].str.lower()
    combined_df['Data'] = combined_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in str(x).split()]))
    combined_df['Data'] = combined_df['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    combined_df['Data'] = combined_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))

    (train, test) = train_test_split(combined_df, test_size=0.2, random_state=42, shuffle=True)
    (train, val) = train_test_split(train, test_size=0.2, random_state=42, shuffle=True)

    train_sentences = train['Data'].to_numpy()
    train_labels = train['Label'].to_numpy()
    test_labels = test_df['Label'].to_numpy()
    
    vocab_size = 10000
    oov_token = "<oov>"

    tokeniser = Tokenizer(num_words = vocab_size,oov_token = oov_token)
    tokeniser.fit_on_texts(train_sentences)
    word_index = tokeniser.word_index
    sequences = tokeniser.texts_to_sequences(train_sentences)
    padding = pad_sequences(sequences,maxlen=120,truncating='post')
    
    test_sentences = test_df['Data'].to_numpy()
    testing_sequences = tokeniser.texts_to_sequences(test_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=120,truncating='post')
    model.fit(testing_padded, test_labels, epochs=10, callbacks=[cp_callback])

if __name__ == "__main__":
    predict_scores(sys.argv[1])