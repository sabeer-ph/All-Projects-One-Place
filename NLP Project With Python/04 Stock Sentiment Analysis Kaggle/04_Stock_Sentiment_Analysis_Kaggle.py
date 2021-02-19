#!/usr/bin/env python
# coding: utf-8

# ## Stock Sentiment Analysis using News Headlines

# In[1]:


import pandas as pd
from IPython.display import display


# In[2]:


df=pd.read_csv('./Data/Stock_News_Data.csv', encoding = "ISO-8859-1")


# In[3]:


print(df.shape)
display(df.head())


# In[4]:


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[6]:


# Removing punctuations
data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True) # replace everython else other than text to " " space

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index # replace column names to 0 to 25.
display(data.head(5))



# In[7]:


# Convertng headlines to lower case
for index in new_Index:
    data[index]=data[index].str.lower() # convert every cell into lower
data.head(2)


# In[7]:


' '.join(str(x) for x in data.iloc[1,0:25]) # test to convert one row all cell into one sentence. 


# In[8]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[9]:


headlines[0]


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[11]:


## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)
traindataset.shape


# In[12]:


# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[13]:


## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[14]:


## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[15]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)


# In[ ]:




