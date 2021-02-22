# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 02:46:30 2021

@author: Sabeer PH

Recommendation systems are a type of information filtering systems because they improve the quality of search results and provide elements that are more relevant to the search item or that are related to the search history of the user.

These are active information filtering systems that personalize the information provided to a user based on their interests, relevance of the information, etc. Recommendation systems are widely used to recommend movies, items, restaurants, places to visit, items to buy, etc.

There are two types of recommendation systems:

1. Content-based filtering
2. Collaborative filtering (picture in the current folder)


The dataset I’ll be using here consists of restaurants in Bangalore, India, collected from Zomato
link = https://www.kaggle.com/himanshupoddar/zomato-bangalore-restaurants/download
"""

# importing the necessary Python Libraries here
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# load datataset 

zomato_real = pd.read_csv('zomato.csv')
zomato_real.head()
zomato_real.columns

#Now the next step is data cleaning and feature engineering for this step we need to do a lot of stuff with the data such as:

#1. Deleting Unnnecessary Columns
zomato = zomato_real.drop(['url','dish_liked','phone'],axis=1)
zomato.columns


#2. Remove Duplicates
zomato.duplicated().sum()
zomato.drop_duplicates(inplace=True)


#3. Remove NAN values
zomato.info()
zomato.isnull().sum()
zomato.dropna(how='any',inplace=True)


#4. Changing the column names
zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type', 'listed_in(city)':'city'})


#5. Some Transformations # rates has comma instead of dots. so we convert it to string --> replace --> convert back to float
zomato['cost'] = zomato['cost'].astype(str) #Changing the cost to string
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.')) #Using lambda function to replace ',' from cost
zomato['cost'] = zomato['cost'].astype(float)


zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

#6 # Adjust the column names
zomato.name = zomato.name.apply(lambda x: x.title()).head()
zomato.online_order.value_counts()
zomato.online_order.replace(('Yes','No'),(True,False),inplace=True)
zomato.book_table.value_counts()
zomato.book_table.replace(('Yes','No'),(True,False),inplace=True)


## Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0


for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato.name == restaurants[i]] = zomato['rate'][zomato.name == restaurants[i]].mean()


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (1,5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

#Now the next step is to perform some text preprocessing steps which include:
    
## 1. Lower Casing
zomato["reviews_list"] = zomato["reviews_list"].str.lower()

## 2. Removal of Puctuations
import string
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))


## 3.Removal of Stopwords
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))


## 4. Removal of URLS
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

zomato[['reviews_list', 'cuisines']].sample(5)



# RESTAURANT NAMES:
restaurant_names = list(zomato['name'].unique())
def get_top_words(column, top_nu_of_words, nu_of_word):
    vec = CountVectorizer(ngram_range= nu_of_word, stop_words='english')
    bag_of_words = vec.fit_transform(column)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:top_nu_of_words]
    
zomato=zomato.drop(['address','rest_type', 'type', 'menu_item', 'votes'],axis=1)
zomato.columns

# Randomly sample 60% of your dataframe
df_percent = zomato.sample(frac=0.5)


# TF-IDF

df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend(name, cosine_similarities = cosine_similarities):
    
    # Create a list to put top restaurants
    recommend_restaurant = []
    
    # Find the index of the hotel entered
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with a similar cosine-sim value and order them from bigges number
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with a similar cosine-sim value
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Creating the new data set to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines','Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines','Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS LIKE %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new
recommend('Pai Vihar')