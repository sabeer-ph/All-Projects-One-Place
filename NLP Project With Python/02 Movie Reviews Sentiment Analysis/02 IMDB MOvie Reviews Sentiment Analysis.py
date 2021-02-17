# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:10:41 2021

@author: Sabeer PH


Project : Movie Reviews Sentiment Analysis (IMDB dataset)

In this Machine Learning Project,
 we’ll build binary classification that puts movie reviews texts into one of two categories 
 — negative or positive sentiment. 
 We’re going to have a brief look at the Bayes theorem and relax its requirements 
 using the Naive assumption.
"""

import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle


#down dataset from https://thecleverprogrammer.com/wp-content/uploads/2020/05/IMDB-Dataset.csv
data = pd.read_csv('IMDB-Dataset.csv')
print(data.shape)
print(data.head())

print(data.info()) # there is no null values 

data.sentiment.value_counts() # balanced dataset with 25000 for postive and negative

# replace Postive with 1 and negative with 0

data.sentiment.replace('positive',1,inplace=True)
data.sentiment.replace('negative',0,inplace=True)

data.review[0]

'''STEPS TO CLEAN THE REVIEWS :
1. Remove HTML tags
2. Remove special characters
3. Convert everything to lowercase
4. Remove stopwords
5. Stemming '''


# 1. Remove HTML tags - Regex rule : ‘<.*?>’

def clean(text):
    cleaned = re.compile('<.*?>')
    return re.sub(cleaned,'',text)

print(clean('<head>this is sample<!head>')) # the tags along the text will be removed.

data.review = data.review.apply(clean)
data.review[0]


#2. Remove special characters
def is_special(text):
    res = ''
    for i in text:
        if i.isalnum():
            res+= i
        else:
            res+= ' '
    return res

is_special('abcd,.;;') # test the above function

data.review = data.review.apply(is_special)
data.review[0]

#3. Convert everything to lowercase

def to_lower(text):
    return text.lower()

to_lower('ABCDegfHIJK') # test the above function


data.review = data.review.apply(to_lower)
data.review[0]


#4. Remove stopwords
def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words]

rem_stopwords('This is the end hold my hands and count to 10')# test the above function

data.review = data.review.apply(rem_stopwords)
data.review[0]

#5. Stem the words - using SnowballStemming
def stem_txt(words):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(word) for word in words])

stem_txt(rem_stopwords('This is the end hold my hands and count to 10'))# test the above function


data.review = data.review.apply(stem_txt)
data.review[0]

data.head()

#CREATING THE MODEL
#1. Creating Bag Of Words (BOW)
''' assign X and y values, instatiate a cv using 1000 features and do fit transform'''

X = data.review
y = data.sentiment
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(X).toarray()

print('X shape :',X.shape)
print('y shape :',y.shape)

#2. Train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=9)
print('X train size : {}, y train size : {}'.format(X_train.shape,y_train.shape))
print('X test size : {}, y test size : {}'.format(X_test.shape,y_test.shape))

#3. Defining the models and Training them
gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=0.1,fit_prior=True),BernoulliNB(alpha=0.1,fit_prior=True)

gnb.fit(X_train,y_train)
mnb.fit(X_train,y_train)
bnb.fit(X_train,y_train)


#4. Prediction and accuracy metrics to choose best model

ygnb = gnb.predict(X_test)
ymnb = mnb.predict(X_test)
ybnb = bnb.predict(X_test)

# test the accuracy score
print(accuracy_score(ygnb,y_test))
print(accuracy_score(ymnb,y_test))
print(accuracy_score(ybnb,y_test))

# dump the model to a pickle file

pickle.dump(bnb,open('model1.pkl','wb'))


# lets table the review to predict
rev =  """Terrible. Complete trash. Brainless tripe. Insulting to anyone who isn't an 8 year old fan boy. Im actually pretty disgusted that this movie is making the money it is - what does it say about the people who brainlessly hand over the hard earned cash to be 'entertained' in this fashion and then come here to leave a positive 8.8 review?? Oh yes, they are morons. Its the only sensible conclusion to draw. How anyone can rate this movie amongst the pantheon of great titles is beyond me.

So trying to find something constructive to say about this title is hard...I enjoyed Iron Man? Tony Stark is an inspirational character in his own movies but here he is a pale shadow of that...About the only 'hook' this movie had into me was wondering when and if Iron Man would knock Captain America out...Oh how I wished he had :( What were these other characters anyways? Useless, bickering idiots who really couldn't organise happy times in a brewery. The film was a chaotic mish mash of action elements and failed 'set pieces'...

I found the villain to be quite amusing.

And now I give up. This movie is not robbing any more of my time but I felt I ought to contribute to restoring the obvious fake rating and reviews this movie has been getting on IMDb."""

# repeat all the process we did
f1 = clean(rev)
f2 = is_special(f1)
f3 = to_lower(f2)
f4 = rem_stopwords(f3)
f5 = stem_txt(f4)

bow,words = [],word_tokenize(f5)
for word in words:
    bow.append(words.count(word))
    
word_dict = cv.vocabulary_ # has a dictionary of word : count after we trained it
pickle.dump(word_dict,open('bow.pkl','wb'))

print(type(word_dict))
print(f5.count('a'))

inp = []
for i in word_dict:
    inp.append(f5.count(i[0]))
    
ypred = bnb.predict(np.array(inp).reshape(1,1000))

print(ypred) # gives 0 which means negative review