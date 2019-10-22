'''
Created on 13 Oct 2017

@author: Gergely
'''
import random as rn


class DataGenerator:
    '''
    generate data for K-means clustering algorithm to work with
    '''

    def __init__(self, centroids, noise, points_per_centroid):
        '''
        centroids-array of centroids(those represented as arrays also)
        noise -the radius of the circle we are generating points inside, which is also distorted in the algorithm to appear more random
        pointsPerCentroid-number of points to generate around each centroid(this is also distorted under the name of randomness)
        '''
        self.centroids = centroids
        self.noise = noise
        self.data = []
        self.points_per_centroid = points_per_centroid

    def generate(self):
        for centroid in self.centroids:
            mistake = self.points_per_centroid + rn.randint(0, 25)
            for p in range(mistake):
                point = []
                for i in range(len(centroid)):
                    val = self.noise / 4
                    distortion = self.noise + rn.random() * (val * 2) - val
                    elem = centroid[i] + rn.random() * (2 * distortion) - distortion
                    point.append(elem)
                self.data.append(point)
        rn.shuffle(self.data)
        return self.data

    def write_to_file(self, fname):
        with open(fname, 'w')as f:
            for point in self.data:
                s = ''
                for elem in point:
                    s += str(elem) + ' '
                s += '\n'
                f.write(s)
