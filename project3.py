#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:53:25 2020

@author: dogacanyilmaz
"""



import sys
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


file1=sys.argv[1]


#Read csv
data=pd.read_csv(file1, sep=',',header=None)
#Set colnames from row 0
colnames=data.loc[0,:].copy()
data.columns=colnames

#drop row 0which is colnames
data.drop(0,axis=0,inplace=True)
#Drop column 0which is product code 
data.drop('Product_Code',axis=1,inplace=True)
#Drop min max and normalized columns
data.drop(colnames[53:107],axis=1,inplace=True)
#Reset index
data.reset_index(inplace=True,drop=True)

#Datatype is string now
#Convert them to int
colnames=data.columns
for i in colnames:
    data[i] = pd.to_numeric(data[i])

#Train set is first 51 weeks
#does not include last week
#it has 51 columns
train=data.iloc[:,0:51]

#Testlabel set is the last week
#We only use it to calculate MSE
testlabel=data.iloc[:,51]




##############################
#We are going to use kernelized ridge regression with LSTM encoding
#Best MSE I acheieved is when:
#Windowsize is 22 
#Alpha is 2


#This function takes a series and returns a LSTM encoded dataframe 
def LSTMencoder(ts,window):
    #Size of this dataframe
    newdf = pd.DataFrame(np.zeros((len(ts)-window, window)))
    
    #copy indexes
    indexes=ts.index.copy()
    count=0
    #For each row of new LSTM encoded dataframe
    for r in range(len(newdf)):
        #For column row of new LSTM encoded dataframe
        for c in range(len(newdf.columns)):
            #Copy that element from ts
            newdf.loc[r,c]=ts[indexes[count]].copy()
            #Increase count
            count+=1
        #Adjust count at the beginning of new row
        count=r+1
    return newdf








#There are the best parameters
bestwindow,bestalpha=22,2

#This will hold predictions to calculate MSE
predictions=[]

#For each product
for i in train.index:
    #print(i)
    #Get series of product i from training
    series=train.loc[i,:].copy()
    #Reset indexes or it causes problems in LSTM encoder
    series.reset_index(inplace=True,drop=True)
    
    #Get LSTM encoded dataframe given windowsize
    dftrain=LSTMencoder(series,bestwindow)
    
    #Test set if only last values of lengt windowsize
    #It does not have anything with last week 
    #because series does not contain week to be predicted
    dftest=series.tail(bestwindow).copy().to_frame().transpose()
    
    #Get trainlabels 
    trainlabel=series.iloc[bestwindow:]

        
    #Fit a standart scaler using train data
    scaler = preprocessing.StandardScaler().fit(dftrain)
    
    #Get standardized train and test data
    dftrain=scaler.transform(dftrain)
    dftest=scaler.transform(dftest)
    
    #Kernelized Ridge Regression
    clf=KernelRidge(alpha=bestalpha, kernel='poly', degree=3)
    
    #Fit the data
    clf.fit(dftrain,trainlabel)
    
    #Predict with test instance for the last week
    prediction = clf.predict(dftest)
    #right now prediction is array with a single element
    #If we print it without this it will have square brackets
    prediction=prediction[0]
    
    #Print the prediction
    print(prediction)
    
    #Record the prediction
    predictions.append(prediction)
    

#This is the first time we are using testlabels(Last week)    
print(mean_squared_error(testlabel,predictions))
