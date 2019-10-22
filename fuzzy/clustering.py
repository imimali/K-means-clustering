'''
    created on 21 October 2019
    
    @author: Gergely
'''

import random as rn

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from data_generator import DataGenerator


def initialize_memberships(nr_centroids, nr_examples):
    return [softmax([rn.random() for _ in range(nr_centroids)])
            for _ in range(nr_examples)]


def compute_centroids(xs, memberships, m):
    new_centroids = [[] for _ in range(len(memberships[0]))]
    for i in range(len(new_centroids)):
        for j in range(len(xs[0])):
            vij = 0
            for k in range(len(xs)):
                vij += (memberships[k][i] ** m) * xs[k][i]
            vij /= sum([memberships[k][i] for k in range(len(xs))])
            new_centroids[i].append(vij)
    print(np.shape(new_centroids))
    return new_centroids


def compute_centroids2(xs, memberships, m):
    return ([[sum([(x[i] * membership[i]) ** m for i in range(len(x))]) /
              sum([elem ** m for elem in membership])
              for membership in memberships]
             for x in xs])


def compute_distances_from_centroids(xs, centroids):
    return [[sum([(x[i] - centroid[i]) ** 2 for i in range(len(x))]) ** 0.5
             for centroid in centroids]
            for x in xs]


def update_memberships(distances, m):
    new_memberships = []
    for distance_vector in distances:
        new_distance_vector = []
        for distance in distance_vector:
            new_distance = 0
            for other_dist in distance_vector:
                new_distance += (distance / other_dist) ** 2
            new_distance_vector.append(new_distance ** (1 / (m - 1)))
        new_memberships.append(new_distance_vector)
    return new_memberships


def fuzzy_k_means(xs, nr_centroids, m=2, max_iter=30, epsilon=1e-2):
    memberships = initialize_memberships(nr_centroids, len(xs))
    centroids = compute_centroids(xs, memberships, m)
    print('centroids', np.shape(centroids), centroids)
    centroid_history = [centroids]
    for _ in range(max_iter):
        print('one step is done')
        distances = compute_distances_from_centroids(xs, centroids)
        centroids = compute_centroids(xs, memberships, m)
        memberships = update_memberships(distances, m)
        centroid_history.append(centroids)
    return centroids, memberships, centroid_history


if __name__ == '__main__':
    xs = np.random.random_integers(100, size=(100,))
    ys = np.random.random_integers(100, size=(100,))
    gen = DataGenerator([[20, 20], [100, 100]], 30, 50)
    data = gen.generate()
    cents, membs, history = fuzzy_k_means(data, 2, max_iter=7)
    print(cents, membs)
    print("*" * 10)
    print(np.shape(cents), np.shape(membs))
    print("*" * 10)
    # for h in history:
    #    print(h)
    data = np.array(list(zip(*data)))
    plt.scatter(data[0], data[1], label='skitskat', color='k')
    plt.show()
