# in utils.py I have:
# def lr_results(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
#         C=2**np.arange(-8, 1).astype(np.float), seed=42):
#     scores = []
#     for i, c in enumerate(C):
#         clf = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
#         clf.fit(trX, trY)
#         score = clf.score(vaX, vaY)
#         scores.append(score)
#     c = C[np.argmax(scores)] 
#     clf = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
#     clf.fit(trX, trY)
#     return clf

# Then transform, train, and save the model:
# from utils import sst_binary, lr_results
# from encoder import Model

# model = Model()

# trX, vaX, teX, trY, vaY, teY = sst_binary()
# trXt = model.transform(trX)
# vaXt = model.transform(vaX)
# teXt = model.transform(teX)

# clf = lr_results(trXt, trY, vaXt, vaY, teXt, teY)

# from joblib import dump, load
# dump(clf, 'logregress_clf.joblib')

import warnings
warnings.filterwarnings('ignore')

from joblib import load
from encoder import Model

# load the trained model
clf = load('logregress_clf.joblib')
model = Model()

import pandas as pd
import tweepy as tw
import re

# Authenticate to Twitter
consumer_key = "1KbEaA98iLFrLI68cWwZBojKa"
consumer_secret = "UMuEh1JiSu8mVtOpWnDKatg2tXCEf3RGrOyUNBPfC0NgPgUVz0"
access_token = "786958989347033088-k12TcKdTNLTcr16b9W2eAioz07eAwu1"
access_token_secret = "3sU3lCoT6b6Y1Gf1DZPENCaYbbd5B18Dkp8iSYNuLWCSV"

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth)

# try:
#     api.verify_credentials()
#     print("Authentication OK")
# except:
#     print("Error during authentication")

# Harvard Business Review
tag = "@HarvardBiz"

tweets = api.search(q=tag, lang="en", count=10)
# tweets_text = [tweet.text for tweet in tweets]

# remove url in tweets
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

# list of tweets strings
clean_tweets = [remove_url(tweet.text) for tweet in tweets]

# transform the tweets list
tw_transfrom = model.transform(clean_tweets)
# use clf to predict for Boolean results
predictions = clf.predict(tw_transfrom)

# print the final results
# an empty line formatting
print()
print("Search tag:", tag)
for i in range(0, 10):
    tw1 = clean_tweets[i]
    sa = predictions[i]
    print(sa, tw1)
