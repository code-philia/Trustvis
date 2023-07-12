"""The Sampeling class serve as a helper module for retriving subject model data"""
from abc import ABC, abstractmethod

import os
import gc
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma
import math
from pynndescent import NNDescent

"""
DataContainder module
1. calculate information entropy for singel sample and subset
2. sample informative subset
"""
class SampelingAbstractClass(ABC):
    
    def __init__(self, data_provider, model, epoch):
        self.mode = "abstract"
        self.data_provider = data_provider
        self.model = model
        self.epoch = epoch
        
    @abstractmethod
    def info_calculator(self):
        pass

class Sampleing(SampelingAbstractClass):
    def __init__(self,data_provider, model, epoch, device):
        self.data_provider = data_provider
        self.model = model
        self.epoch = epoch
        self.DEVICE = device

    def probability_density_cal(self,X,dim,k):
        """
        calculate the probability of each sample
        :param data: numpy.ndarray
        :param k: nearest neibour number
        :return: probability, numpy.ndarray
        """
        ## p(xi) = k / (n * V * σ)
        # step one: calculate volumes:  V = π^(d/2) / Γ(d/2 + 1) * r^d 
        print("start calculate the nbrs")
        # nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
        # distances, indicates = nbrs.kneighbors(X)

        """
        construct a vietoris-rips complex
        """
        # number of trees in random projection forest
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(X.shape[0]))))
        # distance metric
        metric = "euclidean"
        # get nearest neighbors
        
        nnd = NNDescent(
            X,
            n_neighbors=k,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        indicates, distances = nnd.neighbor_graph
        print("finish calculate the nbrs")
        pred = self.data_provider.get_pred(self.epoch,X)
        d = dim  # dimensional
        volumes = []
        variances = []
        r_col = []
        print("start calculate the volumes and variances")
        for i in range(len(X)):
            r = distances[i, -1]  # farest neibour 's distance as r
            V = (math.pi**(d/2) / gamma(d/2 + 1)) * (r**d)  # calculate the volumes
            volumes.append(V)
            r_col.append(r)
            # calculate prediction varians: σ
            neighbor_indices = indicates[i]
            neighbor_preds = pred[neighbor_indices]  # get predictions of k neighbors
            # Flatten the neighbor_preds array to a 1D array
            flatten_preds = neighbor_preds.flatten()

            # Calculate the variance for all predictions
            variance = np.var(flatten_preds)

            # variance = np.var(neighbor_preds, axis=0) # calculate variance for each class
            variances.append(variance)

        print("finsih calculate the volumes and variances")
        # step three: calculate the probability: p(xi) = k / (n * V * σ)
        n = len(X)
        probabilities = []
        for i in range(len(X)):
            # p = k / (n * volumes[i] * variances[i])
            p = k / (n * (volumes[i]) * (variances[i]))
            probabilities.append(p)
        
        return probabilities,volumes,variances,r_col


    def info_calculator(self):
        data = self.data_provider.train_representation(self.epoch)
        # data_nbrs = NearestNeighbors(n_neibour = )

