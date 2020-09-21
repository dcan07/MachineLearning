#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:03:40 2020

@author: dogacanyilmaz
"""

'''
# Local filepaths
datafile='/Users/dogacanyilmaz/Downloads/ionosphere/ionosphere.data'
labelfile='/Users/dogacanyilmaz/Downloads/ionosphere/ionosphere.trainlabels.0'
'''
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


############################
#temp data
'''
data=[[1,1],
      [1,2],
      [1,3],
      [3,1],
      [3,2],
      [3,3],
      [50,2]]
rows=len(data)
cols=len(data[0])
trainlabels={0:0,
             1:0,
             2:0,
             3:1,
             4:1,
             5:1,
             6:1}
n=[3,4]
'''




##############################

eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
#stoppingcondition=0.001
stoppingcondition=0.000000001

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
def hingeloss(dataframe,labels,weight):
    loss=0
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
objective=hingeloss(data,trainlabels,w)

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
    
    
    
    #Adaptive eta part
    best_obj=objective+100000
    for k in range(0, len(eta_list), 1):
        
        eta = eta_list[k]
  
        ##update w
        #We will inverse this update after we choose best_eta
        for j in range(cols):
            w[j]=w[j]-eta_list[k]*gradient[j]
        

        ##get new error
        obj = objective=hingeloss(data,trainlabels,w)

        ##update bestobj and best_eta
        if(obj<=best_obj):
            best_obj=obj
            best_eta=eta_list[k]
        
        
        #Cancel the previous update of w
        for j in range(cols):
            w[j]=w[j]+eta_list[k]*gradient[j]

    eta = best_eta
    print(eta)
    
    
    
    #Update weights
    for j in range(cols):
        w[j]=w[j]-eta*gradient[j]
        
    #Recalculate objective
    objective=hingeloss(data,trainlabels,w)

'''
#Print the w
temp=0
for j in range(cols-1):
    print(w[j])
    temp+=w[j]**2

print(abs(w[cols-1]/(temp**0.5)))
'''


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
            
