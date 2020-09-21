#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:55:59 2020

@author: dogacanyilmaz
"""
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# Local filepaths
file1='/Users/dogacanyilmaz/Dropbox/cs675/spam_train.csv'
file2='/Users/dogacanyilmaz/Dropbox/cs675/spam_test.csv'

#Read csv
train=pd.read_csv(file1, sep=',',header=None,encoding='ISO-8859-1')
test=pd.read_csv(file2, sep=',',header=None,encoding='ISO-8859-1')

#Drop columns 2 to 5
train.drop(train.columns[2:5],axis=1,inplace=True)
test.drop(test.columns[2:5],axis=1,inplace=True)

#Drop first row because it is v1 and v2
train.drop(0,axis=0,inplace=True)
test.drop(0,axis=0,inplace=True)

#Reset index
train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)

#Get train and test label
trainlabel=train.iloc[:,0].copy()
testlabel=test.iloc[:,0].copy()

def preprocesstext(text):
    #Remove the stop words
    stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "","during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    #Regular exp
    text=re.sub('[^a-zA-Z0-9]+', ' ',text)
    #Lower case
    text=text.lower()
    #Get the stemmed text
    text=[nltk.PorterStemmer().stem(w) for w in text.split(" ")] 
    #Remove the stop words
    text = [w for w in text if not w in stop_words] 
    #Remerge the words
    text=" ".join(text)
    return text

#Preprocess the training set
for i in train.index:
    train.iloc[i,1]=preprocesstext(train.iloc[i,1])

#Preprocess the test set
for i in test.index:
    test.iloc[i,1]=preprocesstext(test.iloc[i,1])

#TFIDF vectorizer from sklearn
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)

#Get the TFIDF train and test sets
train_vectors = vectorizer.fit_transform(train.iloc[:,1])
test_vectors = vectorizer.transform(test.iloc[:,1])

#Build the svm
model = svm.LinearSVC(C=1)
#fit the data
model.fit(train_vectors, train.iloc[:,0]) 
#Predict
prediction = model.predict(test_vectors)

#Calculate accuracy
accuracy=0
for i in range(len(prediction)):
    print(prediction[i])
    if prediction[i]==test.iloc[i,0]:
        accuracy+=1
print(accuracy*100/len(prediction))

