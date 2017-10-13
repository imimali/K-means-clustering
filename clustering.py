'''
Created on 4 Oct 2017

@author: Gergely
'''


import random as rn
class KMeansClustering():
    def __init__(self,noOfCentroids,maxIter):
        self.noOfCentroids=noOfCentroids
        self.maxIter=maxIter
        self.centroids={i:[]for i in range(noOfCentroids)}
        
    def computeCentroids(self,X,idx):
        candidates={i:[]for i in range(self.noOfCentroids)}
        for i in range(len(idx)):
            candidates[idx[i]].append(X[i])
        for c in candidates.keys():
            newCentroid=[]
            if candidates[c]!=[]:
                for i in range(len(X[0])):
                    elem=0
                    for x in candidates[c]:
                        elem+=x[i]
                    newCentroid.append(elem/len(candidates[c]))#what if this cluster gets wiped out?
                self.centroids[c]=newCentroid
                  
                       
    def initializeCentroids(self,X):
        idx=[i for i in range(self.noOfCentroids)]
        rn.shuffle(idx)
        for i in range(self.noOfCentroids):
            self.centroids[idx[i]]=X[i]
        
        
        
             
            
    def computeDistanceFromCentroid(self,x,centroid):
        dist=0
        for i in range(len(x)):
            dist+=(x[i]-centroid[i])**2
        return dist
    def findClosestCentroids(self,X):
        '''
        return a list of indices the i.th element saying which cluster the i.th example belongs to 
        '''
        idx=[]
        for x in X:
            distances=[]
            for centroid in self.centroids.values():
                distances.append(self.computeDistanceFromCentroid(x, centroid))
            idx.append(distances.index(min(distances)))
        return idx
    
    def predict(self,x):
        dist=[]
        for centroid in self.centroids.values():
            dist.append(self.computeDistanceFromCentroid(x, centroid)) 
        return dist.index(min(dist))
            
        
        
    def learning(self,X):
        self.initializeCentroids(X)
        centroidHistory=[self.centroids]
        for _ in range(self.maxIter):
            idx=self.findClosestCentroids(X)
            print(self.centroids)
            centroidHistory.append(self.centroids)
            self.computeCentroids(X, idx)
        return centroidHistory
    
    def readData(self,fname):
        f=open(fname,'r')
        line=f.readline().strip()
        X=[]
        while line!='':
            dataRow=line.split(' ')
            xRow=[]
            for i in range(len(dataRow)):
                xRow.append(float(dataRow[i])) 
            X.append(xRow)
            line=f.readline().strip() 
            
        f.close()   
        return X
     
  
                     




    
