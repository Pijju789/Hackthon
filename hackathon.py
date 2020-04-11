# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:24:00 2020

@author: mapi8001
"""
import numpy as np 
import pandas as pd 
import spacy
from wordcloud import WordCloud
data=pd.read_csv("C:/Users/mapi8001/Downloads/Train-data.csv")
data.head()
data.describe()
data1=data[['Product Description','Category']].drop_duplicates()
#data['publish_date']=pd.to_datetime(data['publish_date'],format="%Y%M%d")
#data['year']=data['publish_date'].dt.year
#nlp=spacy.load("en_core_web_lg")

import sklearn.feature_extraction.text as text
def get_imp(bow,mf,ngram):
    tfidf=text.CountVectorizer(bow,ngram_range=(ngram,ngram),max_features=mf,stop_words='english')
    matrix=tfidf.fit_transform(bow)
    return pd.Series(np.array(matrix.sum(axis=0))[0],index=tfidf.get_feature_names()).sort_values(ascending=False).head(100)
### Global trends
bow=data['Product Description'].tolist()
total_data=get_imp(bow,mf=5000,ngram=1)
total_data_bigram=get_imp(bow=bow,mf=5000,ngram=2)
total_data_trigram=get_imp(bow=bow,mf=5000,ngram=3)
### Yearly trends
imp_terms_unigram={}
for y in data['Category'].unique():
    bow=data[data['Category']==y]['Product Description'].tolist()
    imp_terms_unigram[y]=get_imp(bow,mf=5000,ngram=1)
imp_terms_bigram={}
for y in data['Category'].unique():
    bow=data[data['Category']==y]['Product Description'].tolist()
    imp_terms_bigram[y]=get_imp(bow,mf=5000,ngram=2)
imp_terms_trigram={}
for y in data['Category'].unique():
    bow=data[data['Category']==y]['Product Description'].tolist()
    imp_terms_trigram[y]=get_imp(bow,mf=5000,ngram=3)


import matplotlib.pyplot as plt
plt.subplot(1,3,1)
total_data.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Unigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,2)
total_data_bigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Bigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)
plt.subplot(1,3,3)
total_data_trigram.head(20).plot(kind="bar",figsize=(25,10),colormap='Set2')
plt.title("Trigrams",fontsize=30)
plt.yticks([])
plt.xticks(size=20)

