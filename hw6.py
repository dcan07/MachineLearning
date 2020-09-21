#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:09:51 2020

@author: dogacanyilmaz
"""

import sys


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
##############################
#Sorts values in column in ascending order
#The same sorting is applied to labels too, so that every row is same 
def sortcol(column,labels):
    #Bubble sort columns and labels
    n = len(column) 
    for i in range(n): 
        for j in range(0, n-i-1): 
            if column[j] > column[j+1] : 
                temp=column[j]
                column[j]=column[j+1]
                column[j+1]=temp
                temp=labels[j]
                labels[j]=labels[j+1]
                labels[j+1]=temp
    return [column,labels]

def partition(column,labels,cutoffvalue):
    

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
    
    return [gini,cutoffvalue]
        
#Get the unique values in array
def getunique(getlist): 
    uniques= [] 
    for i in getlist: 
        if i not in uniques: 
            uniques.append(i) 
    return uniques

#Given a unique values in a column, returns what should be the split values
#that we use in order to find best split value
def decidethesplitvalues(uniquecol):
    splits=[]
    for i in range(len(uniquecol)-1):
        splits.append((uniquecol[i]+uniquecol[i+1])/2)
    return splits

#function get best column to split and splitvalue
def getbestcolumnandsplit(dataframe,label):
    
    #Start with a gini larger than one
    bestgini=2
    
    #Copy the labels because it is dict
    #We dont want to change it
    #Also we need to filter out the test set
    #Careful in here to not to mess up indexes
    
    train=[]
    test=[]
    trainlabel_copy=[]
    #For each row
    for j in range(len(dataframe)):
        if(label.get(j) == 0):
            train.append(data[j])
            trainlabel_copy.append(0)
        elif(label.get(j) == 1):
            train.append(data[j])
            trainlabel_copy.append(1)
        else:
            test.append(data[j])
    
    #For each column
    for i in range(len(train[0])):    
        
                    
        trainlabels_copy=trainlabel_copy.copy()
        
        #sort the columns and labels accordingly
        sortedcols=sortcol([row[i] for row in train],trainlabels_copy)
        
        #get the unique values in sorted column
        uniquesortedcolumn=getunique(sortedcols[0])
        #Get the split values from unique values in column
        splitvalues=decidethesplitvalues(uniquesortedcolumn)
        
        
        
        #for each split value:        
        for j in range(len(splitvalues)):
            
            #Get gini
            gini=partition(sortedcols[0],sortedcols[1],splitvalues[j])
            #Assign new gini if it is best than current best
            if gini[0]<bestgini:
                bestgini=gini[0]
                bestsplitvalue=gini[1]
                bestcolumn=i
    return [bestsplitvalue,bestcolumn]

temp=getbestcolumnandsplit(data,trainlabels)
print(temp[1])
print(temp[0])

