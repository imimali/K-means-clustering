'''
Created on 13 Oct 2017

@author: Gergely
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import *
from dataGenerator import DataGenerator
from clustering import KMeansClustering



clust=KMeansClustering(3,10)
centroids=[[25,25],[100,100],[100,25]]
gen=DataGenerator(centroids,30,70)

X=gen.generate() 
hist=clust.learning(X)
x=[]
y=[]
for point in X:
    x.append(point[0])
    y.append(point[1])
plt.scatter(x,y,label='skitskat',color='k')
for centroids in hist:    
    for centroid in centroids.values():
    
        plt.scatter([centroid[0]],[centroid[1]],label='skitskat',color='r',marker=11) 
plt.show()