from sklearn.pipeline import Pipeline
import os
# import kaggle
import requests
import urllib.request
import wget
import pandas as pd
import re

# kaggle.api.authenticate() # for this to work, add the kaggle.json file to /Users/xxx/.kaggle folder
# kaggle.api.dataset_download_files('arkhoshghalb/twitter-sentiment-analysis-hatred-speech', path='./data/kaggle_twitter', unzip=True)

if not os.path.exists("data/github_reddit/train.csv"):
    os.mkdir("data/github_reddit")
    url = 'https://raw.githubusercontent.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/master/data/reddit.csv'
    wget.download(url,"data/github_reddit/train.csv")


if not os.path.exists("data/github_twitter/train.csv"):
    os.mkdir("data/github_twitter")
    url = 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv'
    wget.download(url,"data/github_twitter/train.csv")

"""## Cleaning Sourced Data"""

gt_df = pd.read_csv("data/github_twitter/train.csv")
gr_df = pd.read_csv("data/github_reddit/train.csv")

gr_df['hate_speech_idx'] = gr_df['hate_speech_idx'].fillna(0)
data = list()
label = list()
for index, row in gr_df.iterrows():
    if row["hate_speech_idx"] != 0:
        ini_list = row["hate_speech_idx"]
        res = ini_list.strip('][').split(', ')
        res = [int(item) - 1 for item in res]
        temp = row["text"].replace("\t","").split("\n")
        main = [x[x.find('.') + 2:] for x in temp]
        main = main[:-1]
        main = [x.strip("'") for x in main]
        main = [x.strip('"') for x in main]
        main = [x.lstrip('>') for x in main]
        res = [x for x in res if x < len(main)]
        data = data + [main[x] for x in res]
        label = label + [1 for x in res]
        notres = [x for x in range(len(main)) if x not in res]
        data = data + [main[x] for x in notres]
        label = label + [0 for x in notres]
        
dictionary = {'Label':label, 'Data':data}
train_data1 = pd.DataFrame(dictionary)
if not os.path.exists("data/clean"):
    os.system("mkdir data/clean")
train_data1.to_csv("data/clean/reddit.csv")

clean_tweets = []
clean_labels = []
for index, row in gt_df.iterrows():
  s = re.sub(r'[^a-zA-Z0-9_!@#$%^&*\(\)-= \{\}\[\]:;\"]', '', row['tweet'])
  s = re.sub(r'!+ RT', '', s)
  s = re.sub(r'@.*:','TWUser', s)
  s = re.sub(r'@[a-zA-Z0-9_]*','TWUser', s)
  s = s.strip(" ")
  s = s.strip("\t")
  clean_tweets.append(s)
  if(row['class']==0):
    clean_labels.append(1)
  else:
    clean_labels.append(0)
dictionary = {'Label':clean_labels, 'Data':clean_tweets}
train_data2 = pd.DataFrame(dictionary)

combined = pd.concat([train_data1, train_data2], ignore_index=True)
if not os.path.exists("data/clean"):
    os.system("mkdir data/clean")
combined.to_csv("data/clean/combined.csv")