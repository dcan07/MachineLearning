#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:15:58 2020

@author: dogacanyilmaz
"""

import sys
import random
from sklearn import svm

#This only true when we test with origonal data
#It does not affect the predictions when we deal with random hyperplanes
#it just makes testing and comparison easier
useoriginaldata=False


file1=sys.argv[1]
file2=sys.argv[2]
file3=sys.argv[3]


#Read train data
f=open(file1)
data=[]
i=0
l=f.readline()
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    data.append(l2)
    l=f.readline()
nrows=len(data)
ncols=len(data[0])
f.close()


#read labels
f=open(file2)
trainlabels={}
n=[]
n.append(0)
n.append(0)
l=f.readline()
while(l !=''):
    a=l.split()
    trainlabels[int(a[1])]=int(a[0])
    '''
    if(trainlabels[int(a[1])]==0):
        trainlabels[int(a[1])]=-1
        '''
    l=f.readline()
    n[int(a[0])]+=1

#Read test data

k=int(file3)

#k=10000




##############################

#Returns dot product
def dotproduct(v1,v2):
    dp=0
    for index in range(len(v1)):
        dp=dp+(v1[index]*v2[index])
    return dp

#Returns smallest and largest elements in an array
def getsmallestnadlargest(arr):
    minimum=arr[0]
    maximum=arr[0]
    for index in range(len(arr)):
        if(arr[index]<minimum):
            minimum=arr[index]
        if(arr[index]>maximum):
            maximum=arr[index]
    return minimum,maximum

#Modified version of the function in the course website
def getbestC(train,labels):
                
        random.seed()
        allCs = [.001, .01, .1, 1, 10, 100]
        errors = {}
        for j in range(0, len(allCs), 1):
                errors[allCs[j]] = 0
        rowIDs = []
        for i in range(0, len(train), 1):
                rowIDs.append(i)
        nsplits = 10
        for x in range(0,nsplits,1):  
            
            #### Making a random train/validation split of ratio 90:10
            newtrain = []
            newlabels = []
            validation = []
            validationlabels = []
            random.shuffle(rowIDs) #randomly reorder the row numbers      
            #print(rowIDs[0])
            for i in range(0, int(.9*len(rowIDs)), 1):
                newtrain.append(train[rowIDs[i]])
                newlabels.append(labels[rowIDs[i]])
            for i in range(int(.9*len(rowIDs)), len(rowIDs), 1):
                validation.append(train[rowIDs[i]])
                validationlabels.append(labels[rowIDs[i]])
            

            #print(newtrain.shape)
            #### Predict with SVM linear kernel for values of C={.001, .01, .1, 1, 10, 100} ###
            for j in range(0, len(allCs), 1):
                C = allCs[j]
                clf = svm.LinearSVC(C=C,max_iter=10000)
                clf.fit(newtrain, newlabels)
                prediction = clf.predict(validation)
                
                err = 0
                for i in range(0, len(prediction), 1):
                    if(prediction[i] != validationlabels[i]):
                        err = err + 1
                        
                err = err/len(validationlabels)
                errors[C]+=err
                #print("err=",err,"C=",C,"split=",x)


        bestC = 0
        minerr=1.000
        keys = list(errors.keys())
        for i in range(0, len(keys), 1):
                key = keys[i]
                errors[key] = errors[key]/nsplits
                if(errors[key] < minerr):
                        minerr = errors[key]
                        bestC = key
        
        #columns might differ on different sets
        return bestC,minerr

##############################
#This part is used to create dataset that we use with random hyperplanes
projectedtraindata=[]
projectedtrainlabel=[]
projectedtestdata=[]
#For each row in train data
for i in range(len(trainlabels)):
    temp=[]
    #For each new feature
    for feature in range(k):
        temp.append(1)
    projectedtraindata.append(temp)


#For each row in test data
for i in range(nrows-len(trainlabels)):
    temp=[]
    #For each new feature
    for feature in range(k):
        temp.append(1)
    projectedtestdata.append(temp)
#For each row
for i in range(nrows):
    #Should be in training set
    if(trainlabels.get(i)!= None):
        #Also hold the labels
        projectedtrainlabel.append(trainlabels.get(i))



# Generate new features with random hyperlpanes
#For each new feature we
for feature in range(k):
    
    #get random weights for hyperplane
    w=[]
    for j in range(ncols):
        w.append(random.uniform(-1, 1))

    
    #Pick bias for hyperplane
    projection=[]    
    #For each row
    for i in range(nrows):
        #Should be in training set
        if(trainlabels.get(i)!= None):
            projection.append(dotproduct(w,data[i]))
            
      
    smallest,largest=getsmallestnadlargest(projection)
    w0=random.uniform(largest,smallest)
    
    #We can add bias to the projections, get the sign and add to new dataset
    for i in range(len(projection)):
        projection[i]+=w0
        #If negative sign change it to -1
        if(projection[i]<0):
            projectedtraindata[i][feature]=0
        #else it stays as +1
    
    #Now we calculate new data on the test set
    projection=[]  
    #For each row
    for i in range(nrows):
        #Should be in test set
        if(trainlabels.get(i)== None):
            projection.append(dotproduct(w,data[i])+w0)
    for i in range(len(projection)):
        #If negative sign change it to -1
        if(projection[i]<0):
            projectedtestdata[i][feature]=0
        #else it stays as +1
        
        
############################
#This is only used when we check with original data
#It basically resets the projecteddata to be originaldata
#The name is still projected but it is actually the original
if useoriginaldata==True:
    projectedtraindata=[]
    projectedtrainlabel=[]
    projectedtestdata=[]
    #For each row in train data
    for i in range(len(data)):
        temp=[]
        if(trainlabels.get(i)==None):
            projectedtestdata.append(data[i])
        else:
            projectedtraindata.append(data[i])
            projectedtrainlabel.append(trainlabels.get(i))
        

        
############################
#Now we are done with the feature generation
#Lets start model building
temp=getbestC(projectedtraindata,projectedtrainlabel)
   
#SVM         
clf = svm.LinearSVC(C=temp[0],max_iter=10000)
clf.fit(projectedtraindata,projectedtrainlabel)
prediction = clf.predict(projectedtestdata) 


#Prediction
count=0
for i in range(len(data)):
    if(trainlabels.get(i)== None):
        print(prediction[count],i)
        count+=1


