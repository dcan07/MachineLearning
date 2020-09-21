#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:28:47 2020

@author: dogacanyilmaz
"""
import sys
import random
from numpy import array
from sklearn import svm

#Takes up to 2o min to run
#Structure of code
#1)Read Data
#2)Define Fscore and pearson
#3)univariate selecion of 30 features using Fscore
#4)Multivariate feature selection using svm and 10 fold CV
#5)Using best features build SVM model using all data
#6)Prediction

'''
# Local filepaths
file1='/Users/dogacanyilmaz/Dropbox/cs675/project1/traindata'
file2='/Users/dogacanyilmaz/Dropbox/cs675/project1/trueclass'
file3='/Users/dogacanyilmaz/Dropbox/cs675/project1/testdata'
'''
file1=sys.argv[1]
file2=sys.argv[2]
file3=sys.argv[3]


#Read train data
f=open(file1)
train_data=[]
i=0
l=f.readline()
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    train_data.append(l2)
    l=f.readline()
rows=len(train_data)
cols=len(train_data[0])
f.close()
train_data=array(train_data)


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
f=open(file3)
test_data=[]
i=0
l=f.readline()
while(l != ''):
    a=l.split()
    l2=[]
    for j in range(0,len(a),1):
        l2.append(float(a[j]))
    test_data.append(l2)
    l=f.readline()
f.close()
test_data=array(test_data)


##############################


#Calculates f score given dataset and labels

def Fscore(train,labels):
    fscores=[]
    #loop over features
    for k in range(train.shape[1]):
        #print(k)
        positiveclassmean=0
        negativeclassmean=0
        numberofpositiveclass=0
        numberofnegativeclass=0
        positivediff=0
        negativediff=0
        
        #loop over observations
        for l in range(train.shape[0]):
            #If positive class
            if(labels[l]==1):
               positiveclassmean+=train[l,k] 
               numberofpositiveclass+=1
            else:
                negativeclassmean+=train[l,k]
                numberofnegativeclass+=1
        mean=(positiveclassmean+negativeclassmean)/(numberofpositiveclass+numberofnegativeclass)
        positiveclassmean=positiveclassmean/numberofpositiveclass
        negativeclassmean=negativeclassmean/numberofnegativeclass

        
        
        
        #loop through observations again because we needed class means
        for l in range(train.shape[0]):
            #If positive class
            if(labels[l]==1):
               positivediff=positivediff+((train[l,k]-positiveclassmean)**2)
            else:
                negativediff=negativediff+((train[l,k]-negativeclassmean)**2)
        temp=(((positiveclassmean-mean)**2)+((negativeclassmean-mean)**2))/((positivediff/(numberofpositiveclass-1))+(negativediff/(numberofnegativeclass-1)))
        #print(temp)
        fscores.append(temp)
    return fscores

#Calculates pearson correlation given dataset and labels

def pearson(train,labels):
    
    #we only need correlation coefficient of festures with label
    #we dont need the correlations within features
    pearsoncor=[]
    
    nrows=train.shape[0]
    #Get the label mean and other calculations here
    labelmean=0   
    labeldiff=0
    labeldiffarr=[]
    for l in range(nrows):
        labelmean+=labels[l]
    labelmean=labelmean/nrows
    for l in range(nrows):
        temp=labels[l]-labelmean
        labeldiffarr.append(temp)
        labeldiff+=(temp**2)
    labeldiff=labeldiff**0.5
    #loop over features
    for k in range(train.shape[1]):
        print(k)
        featuremean=0
        featurediff=0
        temp1=0
        #loop over observations
        for l in range(nrows):
            featuremean+=train[l,k]
        featuremean=featuremean/nrows
     
        #loop through observations again 
        for l in range(nrows):
            temp=train[l,k]-featuremean
            temp1+=(temp*labeldiffarr[l])
            featurediff+=(temp**2)
        featurediff=featurediff**0.5
        pearsoncor.append(temp1/(featurediff*labeldiff))

    return array(pearsoncor)


# Selects the ncolumns of the highest importance
def getimportantcolumns(importance,ncolumns):
    importance=list(importance)
    final = [] 
    for i in range(0, ncolumns):  
        maxvalue = -1
        temp=0
        for j in range(len(importance)):      
            if importance[j] > maxvalue: 
                maxvalue = importance[j]; 
                temp=j
        final.append(temp) 
        importance[temp]=-2
      
    return array(final)


#Get the unique values in array
def getunique(getlist): 
    getlist=list(getlist)
    uniques= [] 

    for i in getlist: 
        if i not in uniques: 
            uniques.append(i) 
    return array(uniques)


#Modified version of the function in the course website
def getbestC(train,labels,colnumber,Cforfeatureselection):
                
        random.seed()
        allCs = [.001, .01, .1, 1, 10, 100]
        accuracy = {}
        for j in range(0, len(allCs), 1):
                accuracy[allCs[j]] = 0
        rowIDs = []
        for i in range(0, len(train), 1):
                rowIDs.append(i)
        nsplits = 10
        returnedcolumns=[]
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
            
            #####
            #Feature selection Part               
            newtrain=array(newtrain)
            validation=array(validation)
            
            clf = svm.LinearSVC(max_iter=1000000,C=Cforfeatureselection)
            clf.fit(newtrain, newlabels)
            
            #Get weights
            temp=clf.coef_
            
            #Get the columns with highest absolute weight
            important_cols=getimportantcolumns(abs(temp[0]),colnumber)
            important_cols=[ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 13, 14, 17, 21]
            #print(important_cols)
            #Add to importznt columns
            for i in range(len(important_cols)):
                returnedcolumns.append(important_cols[i])
            
            #Update dataset
            newtrain=newtrain[:,important_cols]
            validation=validation[:,important_cols]
            #print(newtrain.shape)
            #### Predict with SVM linear kernel for values of C={.001, .01, .1, 1, 10, 100} ###
            for j in range(0, len(allCs), 1):
                C = allCs[j]
                clf = svm.LinearSVC(C=C,max_iter=1000000)
                clf.fit(newtrain, newlabels)
                prediction = clf.predict(validation)
                
                acc = 0
                for i in range(0, len(prediction), 1):
                    if(prediction[i] == validationlabels[i]):
                        acc = acc + 1
                        
                acc = acc/len(validationlabels)
                accuracy[C]+=acc
                #print("err=",err,"C=",C,"split=",x)


        bestC = 0
        maxacc=-1
        keys = list(accuracy.keys())
        for i in range(0, len(keys), 1):
                key = keys[i]
                accuracy[key] = accuracy[key]/nsplits
                if(accuracy[key] > maxacc):
                        maxacc = accuracy[key]
                        bestC = key
        
        #columns might differ on different sets
        return [bestC,maxacc,returnedcolumns]

#Enumerate all parameters
def parameteroptimization(train,labels):
    values=[]
    for i in range(8,16):
        for j in [.01, .1, 1]:
            values1=[]
            values1.append(i)
            values1.append(j)
            print(i,j)
            result=getbestC(train,labels,i,j)
            values1.append(result[1])
            values1.append(result[0])
            values.append(values1)
    return values

#Fscore and pearson gives almost similar columns
fscore=Fscore(train_data,trainlabels)

#get the most important 30 columns with fscore
important_columns = getimportantcolumns(fscore,30)

#I did some experiments with parameters to find out best combination
#This is the best parameters
train_data=train_data[:,important_columns]

#This enumerates parameters to find best
'''
parameters=parameteroptimization(train_data,trainlabels)
'''
#The results of parameters are:
#Number of features used
#Regularization coefficient for feature selection SVM
#Accuracy
#Best prediction SVM regularization coefficient
'''
[8, 0.1, 0.628125, 0.1]
[8, 1, 0.6379999999999999, 0.01]
[9, 0.01, 0.6340000000000001, 0.01]
[9, 0.1, 0.6224999999999999, 0.01]
[9, 1, 0.6236249999999999, 0.01]
[10, 0.01, 0.6249999999999999, 10]
[10, 0.1, 0.6285000000000001, 1]
[10, 1, 0.63575, 10]
[11, 0.01, 0.6367499999999999, 10]
[11, 0.1, 0.6265000000000001, 0.1]
[11, 1, 0.630125, 0.01]
[12, 0.01, 0.63825, 10]
[12, 0.1, 0.638625, 0.01]
[12, 1, 0.6311249999999999, 1]
[13, 0.01, 0.63825, 0.1]
[13, 0.1, 0.6392500000000001, 10]
[13, 1, 0.6525000000000001, 0.1]
[14, 0.01, 0.6533749999999999, 0.01]
[14, 0.1, 0.6540000000000001, 1]
[14, 1, 0.6419999999999999, 0.1]
[15, 0.01, 0.651375, 0.01]
[15, 0.1, 0.6530000000000001, 10]
[15, 1, 0.647625, 0.1]
'''

#I decided to use these set of parameters 
columnsselected=getbestC(train_data,trainlabels,14,0.1)




bestC=columnsselected[0]
columnsselected=columnsselected[2]



#Lets build the model using all training data and selected columns

#eliminate the duplicated columns used in different validation sets
columnsselected=getunique(columnsselected)

columnsselectedfromoriginadata=important_columns[columnsselected]

#Print the number of columns used
#print(len(columnsselected))
#Print the columns used
print(*columnsselectedfromoriginadata, sep=" ")

#Both below selects the same columns in a different manner
#Select the columns from 30 previously selected columns
train_data=train_data[:,columnsselected]

#Select the columns from original columns
test_data=test_data[:,columnsselectedfromoriginadata]

#Build the model

#clf.fit is not working with dict object
#Make it array
alllabels=[]
for i in range(len(trainlabels)):
    alllabels.append(trainlabels[i])

#model building
clf = svm.LinearSVC(C=bestC,max_iter=1000000)
clf.fit(train_data,alllabels)
prediction = clf.predict(test_data)

#Prediction
for i in range(0, len(prediction), 1):
    print(prediction[i],' ',i)
    



        
