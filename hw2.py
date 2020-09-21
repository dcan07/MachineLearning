
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:44:25 2020

@author: dy234
"""

'''
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
trainlabels={0:-1,
             1:-1,
             2:-1,
             3:-1,
             4:1,
             5:1,
             6:1,
             7:1,}
n=[4,4]
'''




##############################

eta=0.0001
stoppingcondition=0.001


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
objective=0
for i in range(rows):
    if(trainlabels.get(i) != None):
        temp=0
        for j in range(cols):
            temp+=data[i][j]*w[j]
        objective+=(trainlabels[i]-temp)**2
 
iteration=0
previous=objective+10
while(abs(previous-objective)>=stoppingcondition):
    #Assign current objective to be previous objective
    previous=objective
    
    iteration+=1
    #print(iteration)
    #Calculate the gradient
    gradient=[]
    for j in range(cols):
        #this will hold gradient for each column temporarily
        temp=0
        for i in range(rows):
            
            if(trainlabels.get(i) != None):
                #temp1 will hold wTx (including w0)
                temp1=0
                for jprime in range(cols):
                    temp1+=data[i][jprime]*w[jprime]
                temp+=((trainlabels[i]-temp1)*data[i][j])
        #temp=temp*2
        gradient.append(temp)
        
    #Update weights
    for j in range(cols):
        w[j]=w[j]+eta*gradient[j]
        
    #Recalculate objective
    objective=0
    for i in range(rows):
        if(trainlabels.get(i) != None):
            temp=0
            for j in range(cols):
                temp+=data[i][j]*w[j]
            objective+=(trainlabels[i]-temp)**2

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
            
