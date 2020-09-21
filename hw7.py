#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:04:23 2020

@author: dogacanyilmaz
"""

import sys
import random


#Read the data
datafile=sys.argv[1]
f=open(datafile)
data=[]
i=0
l=f.readline()
#read data
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    data.append(l2)
    l=f.readline()
rows=len(data)
cols=len(data[0])
f.close()
#read labels
labelfile=sys.argv[2]
f=open(labelfile)
trainlabels={}
n=[]
n.append(0)
n.append(0)
l=f.readline()
while(l !=''):
    a=l.split()
    trainlabels[int(a[1])]=int(a[0])
    l=f.readline()
    n[int(a[0])]+=1

#Bubble sort columns and labels
#The same sorting is applied to labels too, so that every row is same 
def sortcol(column,lab): 
    
    n = len(column) 
    for i in range(n): 
        for j in range(0, n-i-1): 

            if column[j] > column[j+1] : 
                column[j], column[j+1] = column[j+1], column[j]
                lab[j], lab[j+1] = lab[j+1], lab[j] 
    return [column,lab]


def getginiindex(column,labels,cutoffvalue):

    #array of values in the left and right
    left=[]    
    right=[]
    
    #number of 0 labels in left and right
    lp=0
    rp=0

    
    for c in range(len(column)):
        #If it is in the left partition
        if column[c]<=cutoffvalue:
            #Add to left
            left.append(column[c])
            #if it is label 0
            if labels[c]==0:
                lp+=1
            
        #If it is in the right partition
        else:
            #add to right
            right.append(column[c])
            #if it is label 0
            if labels[c]==0:
                rp+=1
            
    #Calculate the gini based on formula in assignment description
    leftgini=(lp/len(column))*(1-(lp/len(left)))
    rightgini=(rp/len(column))*(1-(rp/len(right)))
    gini=leftgini+rightgini
    
    return gini
        


#Get the unique values in array and after that get the split values
def getsplitvalues(getlist): 
    uniques= [] 
    for i in getlist: 
        if i not in uniques: 
            uniques.append(i) 
    splits=[]
    for i in range(len(uniques)-1):
        splits.append((uniques[i]+uniques[i+1])/2)
    return splits


#function get best column to split and splitvalue
def getbestcolumnandsplit(dataframe,label):
    
    #Start with a gini larger than one
    bestgini=100
    
    
    #For each column
    for i in range(len(dataframe[0])):    
        
        #Work with copies of the objects because they will be changed otherwise
        #sort the columns and labels accordingly
        sortedcols,sortedlabels=sortcol([row[i] for row in dataframe].copy(),label.copy())

        #Get the split values from unique values in column
        splitvalues=getsplitvalues(sortedcols)
        

        
        #for each split value:
        for j in range(len(splitvalues)):
            
            #Get gini
            currentgini=getginiindex(sortedcols,sortedlabels,splitvalues[j])
            #print(currentgini)
            #Assign new gini if it is best than current best
            if currentgini<bestgini:
                bestgini=currentgini
                bestsplitvalue=splitvalues[j]
                bestcolumn=i
    return [bestsplitvalue,bestcolumn]

def bootstrap(dataframe1,label1):
    
    #The input dataset to this function is in the training set
    #No row of test set is not input to this function
    bstrain=[]
    bstrainlabel=[]
    
    #For each row
    for j in range(len(dataframe1)):
    
        #Get a random row index
        #rowindex=numpy.random.randint(0,len(dataframe1))
        rowindex=random.randint(0,len(dataframe1)-1)
        #print(rowindex)

        if(label1[rowindex] == 0):
            bstrain.append(dataframe1[rowindex])
            bstrainlabel.append(0)
        else:
            bstrain.append(dataframe1[rowindex])
            bstrainlabel.append(1)
    #print(len(bstrain))
    return [bstrain,bstrainlabel]
    


#Split the dataset as given
train=[]
test=[]
trainlabel=[]
testindex=[]

#For each row add to splitted sets
for j in range(len(data)):

    if(trainlabels.get(j) == 0):
        train.append(data[j])
        trainlabel.append(0)
    elif(trainlabels.get(j) == 1):
        train.append(data[j])
        trainlabel.append(1)
    else:
        test.append(data[j])
        testindex.append(j)

#Initialize the test_prediction
test_prediction=[]
for j in range(0, len(test)):
    test_prediction.append(0)



#now we start bootstrapping
for k in range(100):
    
    #Bootstap the data
    bstraindata,bstrainlabel=bootstrap(train,trainlabel)
    
    #Get the best split
    splitvalue,splitcolumn=getbestcolumnandsplit(bstraindata,bstrainlabel)
    
    #Lets decide which partition is which label
    leftlabel0=0
    leftlabel1=0
    rightlabel0=0
    rightlabel1=0
    for j in range(len(bstraindata)):
        #if on the left of the split
        if(bstraindata[j][splitcolumn]<=splitvalue):
            if(bstrainlabel[j]==0):
                leftlabel0+=1
            else:
                leftlabel1+=1
        #if on the right split
        else:
            if(bstrainlabel[j]==0):
                rightlabel0+=1
            else:
                rightlabel1+=1
    
    #We got the number of 0 and 1 labels in the both partitions
    
    #if label 0 is larger on the left side and smaller on the right side
    #left side is labeled as 0
    if(leftlabel0>=leftlabel1 and rightlabel0<=rightlabel1):
        for j in range(0, len(test)):
            if(test[j][splitcolumn]<splitvalue): 
                test_prediction[j] += -1
            else:
                test_prediction[j] += 1
    
    #If left side is label 1 and right side is label 0
    else:
        for j in range(0, len(test)):
            if(test[j][splitcolumn]<splitvalue): 
                test_prediction[j] += 1
            else:
                test_prediction[j] += -1
                
    #print(splitcolumn)

#Now print the predictions
for j in range(0, len(test)):	
    if(test_prediction[j] > 0): 
        print("1 ",testindex[j])	
    else:
        print("0 ",testindex[j])

