# First, import the required libraries for the FCM algorithm
"""
@Author: majdi sukkar
@Filename: fcm.py
@Contact: majdiskr@gmail.com
@Time: 2023/07/19 19:55
@Discription: Appearance-Free Post Link
"""
import os
import glob
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import pairwise_distances

class FCM:
    def __init__(self, data, num_clusters=2, max_iters=100, error_threshold=1e-4):
        self.data = data
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.error_threshold = error_threshold
        self.centers = None
        self.memberships = None

    def initialize_membership(self):
        num_samples = self.data.shape[0]
        self.memberships = np.random.rand(num_samples, self.num_clusters)
        self.memberships /= np.sum(self.memberships, axis=1, keepdims=True)

    def update_centers(self):
        power = 2.0 / (self.m - 1)
        self.centers = (self.memberships ** self.m).T.dot(self.data) / np.sum(self.memberships ** self.m, axis=0).reshape(-1, 1)

    def update_membership(self):
        distances = pairwise_distances(self.data, self.centers)
        self.memberships = 1.0 / distances ** (2.0 / (self.m - 1))
        self.memberships /= np.sum(self.memberships, axis=1, keepdims=True)

    def fit(self):
        self.initialize_membership()

        for _ in range(self.max_iters):
            prev_centers = np.copy(self.centers)

            self.update_centers()
            self.update_membership()

            center_shift = np.linalg.norm(self.centers - prev_centers)
            if center_shift < self.error_threshold:
                break

    def predict(self):
        return np.argmax(self.memberships, axis=1)

