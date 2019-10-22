'''
    created on 21 October 2019
    
    @author: Gergely
'''

import random as rn

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from skfuzzy import cmeans

from data_generator import DataGenerator


def initialize_memberships(nr_centroids, nr_examples):
    return [softmax([rn.random() for _ in range(nr_centroids)])
            for _ in range(nr_examples)]


def compute_centroids(xs, memberships, m):
    return ([sum([(x[i] * memberships[i]) ** m] for i in range(len(x))) /
             sum([membership ** m for membership in memberships])
             for x in xs])


def compute_distances_from_centroids(xs, centroids):
    return [[sum([(x[i] - centroid[i]) ** 2] for i in range(len(x))) ** 0.5
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


def fuzzy_k_means(xs, nr_centroids, m, max_iter=30, epsilon=1e-2):
    memberships = initialize_memberships(nr_centroids, len(xs))
    centroids = compute_centroids(xs, memberships, m)
    centroid_history = [centroids]
    for _ in range(max_iter):
        distances = compute_distances_from_centroids(xs, centroids)
        centroids = compute_centroids(xs, memberships, m)
        memberships = update_memberships(distances, m)
        centroid_history.append(centroids)
    return centroids, memberships


if __name__ == '__main__':
    xs = np.random.random_integers(100, size=(100,))
    ys = np.random.random_integers(100, size=(100,))
    print("*" * 10)
    print(initialize_memberships(2, 10))
    print("*" * 10)
    gen = DataGenerator([[20, 20], [100, 100]], 30, 50)
    # data = np.array([xs, ys])
    data = gen.generate()
    data = np.array(list(zip(*data)))
    plt.scatter(data[0], data[1], label='skitskat', color='k')
    print(np.shape(data))
    cntr, u, u0, d, jm, p, fpc = cmeans(data, 2, 2, 1e-2, 1000)
    # print(cntr, u, u0, d, jm, p, fpc)
    print(u)
    plt.show()
