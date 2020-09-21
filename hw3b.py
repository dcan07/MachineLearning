#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:13:39 2020

@author: dogacanyilmaz
"""


import sys
import random

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


#n holds the number of 0 and 1 classes in the train set


eta=float(sys.argv[3])
stoppingcondition=float(sys.argv[4])

C=0.01
#define dot product function
def dotpruduct(a,b):
    dp=0
    if len(a) is len(b):
        for index in range(len(a)):
            dp=dp+a[index]*b[index]
    else:
        print('dot pruduct error')
    return dp
        
# given a weight vector and a dataframe
# calculate the hinge loss function value
def hingeloss(dataframe,labels,weight,cons):
    loss=0
    #This is the hinge part
    for rowindex in range(len(dataframe)):
        if(labels.get(rowindex) != None):
            #temp holds wTx
            temp=dotpruduct(dataframe[rowindex],weight)
            #temp holds 1-y*wTx
            temp=1-(labels.get(rowindex)*temp)
            # max(0,temp)
            if(temp>0):
                # If loss larger than 0, add it to loss 
                loss+=temp
    regloss=0
    #this is the regularization part
    for colindex in range(len(weight)-1):
        regloss+=(weight[colindex]**2)
    regloss=(cons*(regloss**0.5))
    return loss

#This function calculates the gradient using train data
def calculategradient(dataframe,labels,weight):
    dellf=[]
    for colindex in range(len(dataframe[0])):
        dellf.append(0)
        for rowindex in range(len(dataframe)):
            if(labels.get(rowindex) != None):
                # temp holds y*wTx
                temp=dotpruduct(dataframe[rowindex],weight)*labels.get(rowindex)
                if(temp<1):
                    temp1=dataframe[rowindex][colindex]*labels.get(rowindex)*(-1)
                    dellf[colindex]+=temp1
                   
    #This part is the gradients coming from regularization
    #note that we do  add a term for bias because it's not in regularization
    for colindex in range(len(weight)-1):
        dellf[colindex]+=weight[colindex]
    
    return dellf
                    

#Lets change label 0 to label -1
for i in range(rows):
    if(trainlabels.get(i) != None and trainlabels[i]==0):
        trainlabels[i]=-1

# Lets add 1 at the end of each row to calculate w0 easily
for i in range(rows):
    data[i].append(float(1))
cols=len(data[0])

# We need weight vector to be len(col), itialize it
# We can initialize randomly
w=[]
for j in range(cols):
    w.append(0.02*random.random()-0.01)


#lets calculate initial objective value in the training set
objective=hingeloss(data,trainlabels,w,C)

#Lets count the iterations
iteration=0

#We want to enter the main while loop
previous=objective+10
while(abs(previous-objective)>=stoppingcondition):
    #Assign current objective to be previous objective
    previous=objective
    
    # Update iteration count
    iteration+=1
    #print(iteration)
    
    #Calculate the gradient
    gradient=calculategradient(data,trainlabels,w)
  
    #Update weights
    for j in range(cols):
        w[j]=w[j]-eta*gradient[j]
        
    #Recalculate objective
    objective=hingeloss(data,trainlabels,w,C)

###################################################
#Start labeling
for i in range(0,rows,1):
    if(trainlabels.get(i) == None):
        prediction=0
        for j in range(cols,):
            prediction+=data[i][j]*w[j]
        
        
        if(prediction<0):
            print("0 ",i)
        else:
            print("1 ",i)
            
