#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dataset of sms spam - downloaded from UCI
import pandas as pd
from IPython.display import display

data = pd.read_csv('./Data/SMSSpamCollection',sep='\t',header=None,names=['Label','Message'])
data.head()


# In[2]:


## here we will import all packages needed for preprocessing and stopwords
# also first we will be using stemming

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# ### first we try using Stemming

# In[3]:


# institalize stemmer

PS = PorterStemmer()


# In[4]:


# preprocessing starts here
# 1. loop through each sentences
# 2. remove everything other than text
# 3. convert all text to lower
# 4. stip sentence into words - check if its a stop words else applying stemming
# 5. join each word to a sentence back.
# 6. append each sentences to a list
corpus = []
for i in range(len(data['Message'])):
    review = re.sub('[^a-zA-Z]',' ',data['Message'][i])
    review = review.lower()
    review = review.split()
    review = [PS.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
corpus[0]


# In[5]:


# using CountVectorizer we will vectorize the data (Bag of Words)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X.shape)

# the shape of X shows that we have 6296 columns . howevere there will be some words which will be less frequent likes names etc
# so we select only 5000 columns

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
print(X.shape)


# In[6]:


# now that our independent variable X is ready lets work to get the dependent varoable y

display(data.Label.head()) # its a string column and for ML we need to convert it to numerical

label_numerical = pd.get_dummies(data.Label)

display(label_numerical.head()) # 2 columns which is numerical 

# now we just need one column for the ML as label . 
y = label_numerical['spam'].values
display(y)


# In[7]:


# now that our X and y are ready lets train the model with train data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

print(X_train.shape,X_test.shape) # shape of test and train data


# In[8]:


# we will use Naive Bayes model as it works well with NLP. Naive Bayes is a classification model that based on probability

from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)


# In[9]:


# we will check the accuracy of our model
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test,y_pred)


# In[10]:


# also we will check the confusion matrix

confusion_matrix(y_test,y_pred)


# ### using Lemmatization

# In[11]:


# institalize lemmatization

from nltk.stem import WordNetLemmatizer

Lem = WordNetLemmatizer()

# Preprocessing
corpus_lem = []
for i in range(len(data['Message'])):
    review = re.sub('[^a-zA-Z]',' ',data['Message'][i])
    review = review.lower()
    review = review.split()
    review = [Lem.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_lem.append(review)
#corpus[0]

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus_lem).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))


# ### summary :
# 
# We used Bag of words with Stemming and Lemmatized Dataset and we see that the accuracy when used Stemming was better

# ### Test TD-IDF on stemming and lemmatized dataset

# #### 1. stemming

# In[12]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=5000)
X = tv.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))


# #### 2. Lemmatize

# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features=5000)
X = tv.fit_transform(corpus_lem).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))


# #### summary : Accuarcy score reducced when we used TD-IDF with stemming and Lemmatized dataset
