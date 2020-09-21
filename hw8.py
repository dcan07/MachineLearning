#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:29:38 2020

@author: dogacanyilmaz
"""


'''
# Local filepaths
datafile='/Users/dogacanyilmaz/Downloads/ionosphere2/ionosphere.data'
datafile='/Users/dogacanyilmaz/Downloads/climate_simulation2/climate.data'

'''
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
#read number of clusters
numberofclusters=sys.argv[2]
numberofclusters=int(5.0)

# Calculates the k-means objective function given data and clusters
def kmeansobj(dataset,cluster,numofclusters,clustermean):
    kmeansobj=0
    #For each clusters
    for c in range(numofclusters):
        #for each row of data
        for row in range(len(dataset)):
            #If row d belings to class c
            if cluster[row]==c:
                #kmeansobj each column
                for col in range(len(dataset[0])):
                    kmeansobj=kmeansobj+((dataset[row][col]-clustermean[c][col])**2)
    return kmeansobj

def calculateclustermeans(dataset,cluster,numofclusters):
    #Initialize both to be 0
    clustermean=[]
    clustersize=[]
    for c in range(numofclusters):
        clustersize.append(0)
        temp=[]
        for col in range(len(dataset[0])):
            temp.append(0)
        clustermean.append(temp)
        
    #Now we can start calculating mean    
    #For each clusters
    for c in range(numofclusters):
        #for each row of data
        for row in range(len(dataset)):
            #If row d belings to class c
            if cluster[row]==c:
                
                #Increase the cluster size
                clustersize[c]+=1
                
                #for each column
                for col in range(len(data[0])):
                    clustermean[c][col]+=dataset[row][col]
    
    #Now divide the sum by size to calculate mean
    for c in range(numofclusters):

        #for each column
        for col in range(len(data[0])):
            clustermean[c][col]/=clustersize[c]
    return clustermean

# assign to the closestcluster
def assigncluster(dataset,numofclusters,clustermean):
    newclusters=[]
    row=0
    c=0
    #for each row of data
    for row in range(len(dataset)):
        #Create the necessary array
        newclusters.append(0)
        
        #make the closest (minimum) distance a very high number inially
        mindist=10000000000000000
        
        #For each clusters we should calculate the distance to cluster mean
        #We will hold these distances in temp
        temp=[]
        for c in range(numofclusters):
            #initialize zero distance
            temp.append(0)
            
            
            
            #for each column
            for col in range(len(dataset[0])):
                    temp[c]+=((dataset[row][col]-clustermean[c][col])**2)
            
            #Take the squareroot of distance
            temp[c]=temp[c]**0.5
            
            #Check if it is the closest cluster center
            if temp[c]<mindist:
                mindist=temp[c]
                newclusters[row]=c
    return newclusters
############################
# Initialize clusters

# First assign the clusters randomly to each row
clusters=[]
for i in range(len(data)):
    clusters.append(random.randint(0,numberofclusters-1))

# Calculate initial cluster means
clustermeans=calculateclustermeans(data,clusters,numberofclusters)

#Calculate initial objective
obj=kmeansobj(data,clusters,numberofclusters,clustermeans)
prevobj=obj+1

i=0

while(abs(prevobj - obj) > 0.000000001):
    
    #Assign previous objective
    prevobj=obj
    
    # Calculate new cluster means
    clustermeans=calculateclustermeans(data,clusters,numberofclusters)
    
    #Assign based on mew cluster means
    clusters=assigncluster(data,numberofclusters,clustermeans)
    
    #Calculate the new objective
    obj=kmeansobj(data,clusters,numberofclusters,clustermeans)
    #print(i)
    i+=1
    



#Now print the clusters
for j in range(0, len(data)):	
    print(clusters[j],j)	


