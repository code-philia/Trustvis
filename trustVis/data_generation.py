"""The Sampeling class serve as a helper module for retriving subject model data"""
from abc import ABC, abstractmethod

import os
import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from singleVis.utils import *
# from sklearn.neighbors import NearestNeighbors
# from scipy.special import gamma
# import math
# from pynndescent import NNDescent
# from sklearn.cluster import KMeans

from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
from pynndescent import NNDescent


"""

"""
class DataGenerationAbstractClass(ABC):
    
    def __init__(self, data_provider, epoch):
        self.mode = "abstract"
        self.data_provider = data_provider
        # self.model = model
        self.epoch = epoch
        
    # @abstractmethod
    # def info_calculator(self):
    #     pass

class DataGeneration(DataGenerationAbstractClass):
    def __init__(self, model, data_provider, epoch, device, n_neighbors=15):
        self.data_provider = data_provider
        self.model = model
        # self.model = model
        self.epoch = epoch
        self.DEVICE = device
        self.n_neighbors = n_neighbors
    
    def get_nearest_n_neighbors(self, train_data, n_neighbors,metric='euclidean'):
        n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
        # max number of nearest neighbor iters to perform
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        # distance metric
        # # get nearest neighbors
        nnd = NNDescent(
            train_data,
            n_neighbors=n_neighbors,
            metric=metric,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        knn_indices, knn_dists = nnd.neighbor_graph

        return knn_indices, knn_dists
    
    def if_border(self,data):
        mesh_preds = self.data_provider.get_pred(self.epoch, data)
        mesh_preds = mesh_preds + 1e-8

        sort_preds = np.sort(mesh_preds, axis=1)
        diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
        border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        border[diff < 0] = 1
        
        return border

    def get_boundary_sample(self):
        """
            Identify the k nearest neighbors for each sample. 
            If any of these neighbors have a prediction differing from the sample, 
            create a boundary sample at the midpoint between the sample and its differing neighbor.
        """
        train_data = self.data_provider.train_representation(self.epoch)
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1])
        pred_res = self.data_provider.get_pred(self.epoch, train_data).argmax(axis=1)

        # find k nearest neibour for each sample
        knn_indices, knn_dists = self.get_nearest_n_neighbors(train_data, self.n_neighbors)

        unique_pairs = set()
        boundary_samples = []

        for i in range(len(train_data)):
            pred_ = pred_res[i]
            k_n_indicates = knn_indices[i]
            k_nn_pred = pred_res[k_n_indicates]

            for j in range(1, len(k_n_indicates)):
                if pred_ != k_nn_pred[j]:
                    # Ensure the smaller index is always first in the pair
                    pair = (min(i, k_n_indicates[j]), max(i, k_n_indicates[j]))
                    # pair = (i, k_n_indicates[j]) if i < k_n_indicates[j] else (k_n_indicates[j], i)

                    # Check if the pair is unique
                    if pair not in unique_pairs:
                        unique_pairs.add(pair)

                        sample1 = train_data[pair[0]]
                        sample2 = train_data[pair[1]]
                        # Calculate the midpoint between the two samples
                        midpoint = (sample1 + sample2) / 2
                        boundary_samples.append(midpoint)

        boundary_samples = np.array(boundary_samples)
        border=self.if_border(boundary_samples)
        boundary_samples = boundary_samples[border==1]
        print("boundary sample sizess", boundary_samples.shape)

        return boundary_samples

        

        
    
    

    







       

