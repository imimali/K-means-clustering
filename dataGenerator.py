'''
Created on 13 Oct 2017

@author: Gergely
'''
import random as rn

class DataGenerator():
    '''
    generate data for K-means clustering algorithm to work with
    '''
    def __init__(self,centroids,noise,pointsPerCentroid):
        '''
        centroids-array of centroids(those represented as arrays also)
        noise -the radius of the circle we are generating points inside, which is also distorted in the algorithm to appear more random
        pointsPerCentroid-number of points to generate around each centroid(this is also distorted under the name of randomness)
        '''
        self.centroids=centroids
        self.noise=noise
        self.data=[]
        self.pointsPerCentroid=pointsPerCentroid
        
    def generate(self):
        for centroid in self.centroids:
            mistake=self.pointsPerCentroid+ rn.randint(0,25)
            for p in range(mistake):
                point=[]
                for i in range(len(centroid)):
                    val=self.noise/4
                    distorsion=self.noise+rn.random()*(val*2)-val 
                    elem=centroid[i]+rn.random()*(2*distorsion)-distorsion
                    point.append(elem)
                self.data.append(point)
        rn.shuffle(self.data)
        return self.data
    def writeToFile(self,fname):
        f=open(fname,'w')
        for point in self.data:
            s=''
            for elem in point:
                s+=str(elem)+' '
            s+='\n'
            f.write(s)
        f.close()

