# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 02:54:58 2021

@author: Sabeer PH

Project : Twitter Sentiment Analysis

"""

# import all Packages Here

import numpy as np # for math calculation
import pandas as pd # for read data and manipulate
from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt

# read the dataset in the current folder

data = pd.read_csv('Sentiment.csv')
print(data.columns)

# we will be needing only 2 columns from this 1. text and 2. sentiment
data = data[['text','sentiment']]

#splitting the data train test set. The test set is the 10% of the original data set.

train,text = train_test_split(data,test_size=0.1)

#For this we drop analysis neutral tweets,
# as my goal was to only differentiate positive and negative tweets.

train.head(10)
train = train[train.sentiment != 'Neutral']
train.head(10)

#separated the Positive and Negative tweets of the training set in order to easily visualize their contained words.

train_pos = train[train.sentiment == 'Positive']['text']
train_neg = train[train.sentiment == 'Negative']['text']

#clean the text from hashtags, mentions and links. 
#Now they were ready for a WordCloud visualization which shows only the most emphatic words of the Positive and Negative tweets.

print(train_pos.head())

def wordcloud_draw(data,color='black'):
    data = ' '.join(data)
    cleaned_word = ' '.join([word for word in data.split() 
                             if 'http' not in word
                             and not word.startswith('@') 
                             and not word.startswith('#') 
                             and word != 'RT'])
    word_cloud = WordCloud(stopwords=STOPWORDS,width=400,height=400,background_color=color).generate(cleaned_word) 
    
    plt.figure(1,figsize=(13,13))
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show
    
print('Positive Words')
wordcloud_draw(train_pos,'white')
print('Negative Words')
wordcloud_draw(train_neg)


#After the vizualization, I removed the hashtags, mentions, links and stopwords from the training set.

#Stop Word: Stop Words are words which do not contain important significance to be used in Search Queries.

#Usually these words are filtered out from search queries because they return vast amount of unnecessary information. ( the, for, this etc. )
print(train.head())

tweets = []  # to add final text and sentiment as a tuple
stopwords_set = set(stopwords.words('english'))

for index,row in train.iterrows():
    cleaned_words = [word.lower() for word in row.text.split() if len(word) > 3]#take words with lenth ge than 3 and lower
    formatted_words = [word for word in cleaned_words 
                       if 'http' not in word
                       and not word.startswith('@')
                       and not word.startswith('#')
                       and word != 'RT']
    words_without_stopwords = [word for word in formatted_words if word not in stopwords_set]
    tweets.append((words_without_stopwords,row.sentiment))
    
print(tweets[0])    
#next step I extracted the so called features with nltk lib, 
#first by measuring a frequent distribution and by selecting the resulting keys.

def get_words_in_tweets(tweets):
    '''get all words as one list'''
    all = []
    for (tweet,sentiment) in tweets:
        all.extend(tweet)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

w_features = get_word_features(get_words_in_tweets(tweets))
#print(w_features)

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#Hereby I plotted the most frequently distributed words. The most words are centered around debate nights.
wordcloud_draw(w_features)

#Using the nltk NaiveBayes Classifier I classified the extracted tweet word features.

# Training the Naive Bayes Classifier

training_set = nltk.classify.apply_features(extract_features,tweets)
print(training_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)


neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))  