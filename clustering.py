'''
Created on 4 Oct 2017

@author: Gergely
'''

import random as rn


class KMeansClustering:
    def __init__(self, nr_centroids, max_iter):
        self.nr_centroids = nr_centroids
        self.max_iter = max_iter
        self.centroids = {i: [] for i in range(nr_centroids)}

    def compute_centroids(self, xs, idx):
        candidates = {i: [] for i in range(self.nr_centroids)}
        for i in range(len(idx)):
            candidates[idx[i]].append(xs[i])
        for c in candidates.keys():
            new_centroid = []
            if candidates[c]:
                for i in range(len(xs[0])):
                    elem = sum([x[i] for x in candidates[c]])
                    new_centroid.append(elem / len(candidates[c]))  # what if this cluster gets wiped out?
                self.centroids[c] = new_centroid

    def initialize_centroids(self, xs):
        idx = [i for i in range(self.nr_centroids)]
        rn.shuffle(idx)
        for i in range(self.nr_centroids):
            self.centroids[idx[i]] = xs[i]

    def compute_distance_from_centroid(self, x, centroid):
        return sum([(x[i] - centroid[i]) ** 2] for i in range(len(x)))

    def find_closest_centroids(self, xs):
        '''
        return a list of indices the i.th element saying which cluster the i.th example belongs to 
        '''
        idx = []
        for x in xs:
            distances = [self.compute_distance_from_centroid(x, centroid)
                         for centroid in self.centroids.values()]
            idx.append(distances.index(min(distances)))
        return idx

    def predict(self, x):
        dist = []
        for centroid in self.centroids.values():
            dist.append(self.compute_distance_from_centroid(x, centroid))
        return dist.index(min(dist))

    def learning(self, xs):
        self.initialize_centroids(xs)
        centroid_history = [self.centroids]
        for _ in range(self.max_iter):
            idx = self.find_closest_centroids(xs)
            print(self.centroids)
            centroid_history.append(self.centroids)
            self.compute_centroids(xs, idx)
        return centroid_history

    def read_data(self, fname):
        with open(fname, 'r')as f:
            line = f.readline().strip()
            X = []
            while line != '':
                data_row = line.split(' ')
                x_row = []
                for i in range(len(data_row)):
                    x_row.append(float(data_row[i]))
                X.append(x_row)
                line = f.readline().strip()

            return X
