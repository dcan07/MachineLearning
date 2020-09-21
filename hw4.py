#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:38:13 2020

@author: dogacanyilmaz
"""


'''
# Local filepaths
datafile='/Users/dogacanyilmaz/Dropbox/cs675/climate_simulation/climate.data'
labelfile='/Users/dogacanyilmaz/Dropbox/cs675/climate_simulation/climate.trainlabels.0'
'''
import sys
import random
import math

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


############################
#temp data
'''
data=[[0,0],
      [0,1],
      [1,0],
      [1,1],
      [10,10],
      [10,11],
      [11,10],
      [11,11]]
rows=len(data)
cols=len(data[0])
trainlabels={0:0,
             1:0,
             2:0,
             3:0,
             4:1,
             5:1,
             6:1,
             7:1}

data=[[1,2],
      [2,1],
      [2,2],
      [2,3],
      [4,1],
      [4,2],
      [4,3],
      [50,2]]
rows=len(data)
cols=len(data[0])
trainlabels={0:0,
             1:0,
             2:0,
             3:0,
             4:1,
             5:1,
             6:1,
             7:1}

n=[4,4]
'''




##############################

eta=float(sys.argv[3])
stoppingcondition=float(sys.argv[4])
'''
eta=0.1
stoppingcondition=0.0000001

eta=0.001
stoppingcondition=0.001
'''
#define dot product function
def dotproduct(a,b):
    dp=0
    if len(a) is len(b):
        for index in range(len(a)):
            dp=dp+a[index]*b[index]
    else:
        print('dot pruduct error')
    return dp

def sigmoid(weights,row):
    return (1/(1+math.exp((-1)*dotproduct(weights,row))))
     
# given a weight vector and a dataframe
# calculate the -loglogistic loss function value
def minuslogloss(dataframe,labels,weight):
    loss=0
    for rowindex in range(len(dataframe)):
        if(labels.get(rowindex) != None):
            #Holds sigmoid value
            sigmoidvalue=sigmoid(weight,dataframe[rowindex])
            if(labels.get(rowindex) == 1):
                loss=loss-(math.log(sigmoidvalue))
            else:
                loss=loss-(math.log(1-sigmoidvalue))
                
            
            #loss=loss-((labels.get(rowindex))*math.log(sigmoidvalue))-((1-labels.get(rowindex))*(math.log(1-sigmoidvalue)))
            
    return loss



#This function calculates the gradient using train data
def calculategradient(dataframe,labels,weight):
    dellf=[]
    for colindex in range(len(dataframe[0])):
        dellf.append(0)
        for rowindex in range(len(dataframe)):
            if(labels.get(rowindex) != None):
                # holds sigmoid value
                sigmoidvalue=sigmoid(weight,dataframe[rowindex])
                dellf[colindex]=dellf[colindex]+((labels.get(rowindex)-sigmoidvalue)*(dataframe[rowindex][colindex]))
                
    return dellf
                    


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
objective=minuslogloss(data,trainlabels,w)

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
        w[j]=w[j]+eta*gradient[j]
        
    #Recalculate objective
    objective=minuslogloss(data,trainlabels,w)
    #print(objective)
    

        

'''
#Print the w
temp=0
for j in range(cols-1):
    print(w[j])
    temp+=w[j]**2

print(w[cols-1]/(temp**0.5))
'''


###################################################
#Start labeling
for i in range(0,rows,1):
    if(trainlabels.get(i) == None):
        prediction=sigmoid(data[i],w)
        if(prediction<0.5):
            print("0 ",i)
        else:
            print("1 ",i)
            
