from sklearn.pipeline import Pipeline
import os

import re
import time
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import nltk
import csv

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
lem = nltk.stem.wordnet.WordNetLemmatizer()
stop_words = nltk.download('stopwords')
tf.random.set_seed(1234)

import string
table = str.maketrans('', '', string.punctuation)
from transformers import BertModel,BertTokenizer,TFBertForSequenceClassification

MAX_LEN = 128
# Create a mask of 1s for each token followed by 0s for padding

def create_masks(inp):
  att_masks=[]
  for seq in inp:
    seq_mask = [float(i>0) for i in seq]
    att_masks.append(seq_mask)
  return att_masks

def calc_accuracy(pred,true_l):
  correct=0
  wrong=0
  for i in range(0,len(pred)):
    if pred[i]==true_l[i]:
      correct+=1
    else:
      wrong += 1
  accuracy = correct/len(pred)
  return accuracy


combined_df = pd.read_csv("data/clean/combined.csv", index_col = "Unnamed: 0")
num_to_sample = np.sum(combined_df['Label']==1)
df_zero = combined_df.query("Label==0").sample(n = num_to_sample, random_state=1)
df_one = combined_df.query("Label==1")
combined_df = df_zero.append(df_one, ignore_index=True)
combined_df = combined_df.sample(frac = 1)

combined_df['Data'] = combined_df['Data'].str.lower()
combined_df['Data'] = combined_df['Data'].apply(lambda x: ' '.join([word.translate(table) for word in str(x).split()]))
#combined_df['Data'] = combined_df['Data'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in stop_words]))
combined_df['Data'] = combined_df['Data'].apply(lambda x: ' '.join([porter.stem(word) for word in x.split()]))

## BERT ##
sentences = combined_df.Data.values


# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

labels = combined_df.Label.values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]


# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels,
                                                            random_state=42, test_size=0.2,shuffle=True)
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_inputs, train_labels,
                                                            random_state=42, test_size=0.2,shuffle=True)
train_masks = create_masks(train_inputs)
test_masks = create_masks(test_inputs)
validation_masks = create_masks(validation_inputs)

train_inputs = tf.convert_to_tensor(train_inputs)
validation_inputs = tf.convert_to_tensor(validation_inputs)
train_labels=tf.convert_to_tensor(train_labels)
validation_labels=tf.convert_to_tensor(validation_labels)
train_masks=tf.convert_to_tensor(train_masks)
validation_masks=tf.convert_to_tensor(validation_masks)
test_inputs = tf.convert_to_tensor(test_inputs)
test_masks = tf.convert_to_tensor(test_masks)
test_labels = tf.convert_to_tensor(test_labels)

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
epochs = 10

n_splits = 10
batch_size = 8
learning_rate = 2e-5
n_validate_per_epoch = 5
optimizer = tf.keras.optimizers.Adam(learning_rate)#,epsilon = 1e-8)
#cat_loss = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits = True)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
since = time.time()
model.fit([train_inputs,train_masks], train_labels,
                  epochs=epochs,verbose=1,batch_size=batch_size,validation_data=([validation_inputs,validation_masks],validation_labels))
end = time.time()

time_taken = end - since

predictions = model.predict([test_inputs,test_masks],batch_size=batch_size)

preds = predictions[:2][0]

logits =[]
pred_labels=[]

for i in preds:
  if(i[1]>=0.5):
    pred_labels.append(1)
  else:
    pred_labels.append(0)
  logits.append(i[1])
acc = calc_accuracy(pred_labels,test_labels)
print("Time taken to train:")
print(time_taken)
print("Accuracy on test set is :")
print(acc)