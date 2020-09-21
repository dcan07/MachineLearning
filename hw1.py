
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:44:25 2020

@author: dy234
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


#n holds the number of 0 and 1 classes in the train set


##############################
#compute the means
#Initialize them to be 0.1???
#label0
m0=[]
for j in range(0,cols,1):
    m0.append(0.1)
#label1
m1=[]
for j in range(0,cols,1):
    m1.append(0.1)


#loop in every row to sum based on classes on the train set
for i in range(0,rows,1):
    #If this row is in the train set with label 0
    if(trainlabels.get(i) != None and trainlabels[i]==0):
        for j in range(0,cols,1):
            m0[j]=m0[j]+data[i][j]
    
    #If this row is in the train set with label 1
    if(trainlabels.get(i) != None and trainlabels[i]==1):
        for j in range(0,cols,1):
            m1[j]=m1[j]+data[i][j]

#get mean by diciding the sum by number of elements
for j in range(0,cols,1):
    m0[j]=m0[j]/n[0]
    m1[j]=m1[j]/n[1]
    
##############################
#compute the variance
#Initialize them to be 0.1???
#label0
s0=[]
for j in range(0,cols,1):
    s0.append(0)
#label1
s1=[]
for j in range(0,cols,1):
    s1.append(0)

#loop in every row to sum based on classes on the train set
for i in range(0,rows,1):
    #If this row is in the train set with label 0
    if(trainlabels.get(i) != None and trainlabels[i]==0):
        for j in range(0,cols,1):
            s0[j]=s0[j]+(data[i][j]-m0[j])**2
    
    #If this row is in the train set with label 1
    if(trainlabels.get(i) != None and trainlabels[i]==1):
        for j in range(0,cols,1):
            s1[j]=s1[j]+(data[i][j]-m1[j])**2

#get variance by diciding the sum by number of elements and take squareroot
for j in range(0,cols,1):
    s0[j]=s0[j]/n[0]
    s0[j]=s0[j]**0.5
    s1[j]=s1[j]/n[1]
    s1[j]=s1[j]**0.5

###################################################
#Start labeling
for i in range(0,rows,1):
    if(trainlabels.get(i) == None):
        d0=0
        d1=0
        for j in range(0,cols,1):
            d0=d0+((m0[j]-data[i][j])/s0[j])**2
            d1=d1+((m1[j]-data[i][j])/s1[j])**2
        if(d0<d1):
            print("0 ",i)
        else:
            print("1 ",i)
            
