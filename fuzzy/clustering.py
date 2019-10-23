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


def compute_centroid3(xs, memberships, m):
    new_centroids = [[] for _ in range(len(memberships[0]))]
    for i in range(len(new_centroids)):
        for j in range(len(xs[0])):
            vij = 0
            for k in range(len(xs)):
                vij += (memberships[k][j] ** m) * xs[k][j]
            vij /= sum([memberships[k][j] ** m for k in range(len(xs))])
            new_centroids[i].append(vij)
    return new_centroids


def compute_centroids(xs, memberships, m):
    new_centroids = [[] for _ in range(len(memberships[0]))]
    for cluster_index in range(len(memberships[0])):
        for i in range(len(xs[0])):
            v = 0
            for k in range(len(xs)):
                v += (memberships[k][cluster_index] ** m) * xs[k][i]
            v /= sum([memberships[k][cluster_index] ** m for k in range(len(xs))])

            new_centroids[cluster_index].append(v)
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
                new_distance += (distance / other_dist) ** (2 / (m - 1))
            new_distance_vector.append(new_distance ** -1)
        new_memberships.append(new_distance_vector)
    return new_memberships


def fuzzy_k_means(xs, nr_centroids, m=2, max_iter=30, epsilon=1e-2):
    memberships = initialize_memberships(nr_centroids, len(xs))
    centroids = compute_centroids(xs, memberships, m)
    centroid_history = [centroids]
    membership_history = [memberships]
    for _ in range(max_iter):
        centroids = compute_centroids(xs, memberships, m)
        distances = compute_distances_from_centroids(xs, centroids)
        memberships = update_memberships(distances, m)
        centroid_history.append(centroids)
        membership_history.append(memberships)
    return centroids, memberships, centroid_history, membership_history


if __name__ == '__main__':
    gen = DataGenerator([[20, 20], [100, 100]], 30, 50)
    data = gen.generate()
    cents, membs, history, memb_hist = fuzzy_k_means(data, 2, max_iter=100)
    print(cents, membs)
    for h in range(len(history)):
        print('*', history[h])
    data = np.array(list(zip(*data)))
    plt.scatter(data[0], data[1], label='skitskat', color='k')
    plt.scatter([cents[0][0], cents[1][0]], [cents[0][1], cents[1][1]], label='skitskat', color='r', marker=11)
    plt.show()
