# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 06:36:15 2021

@author: U277445
"""
from numpy.random import choice
import numpy as np
from sklearn.metrics import euclidean_distances

def exploratory_kmeans_plusplus(n_clusters, data, data_weights, fixed_centroids):
    """
    

    Parameters
    ----------
    n_clusters : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    fixed_centroids : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    n_fixed_centroids , _ = fixed_centroids.shape
    n_points , _ = data.shape
    
    n_new_centroids = n_clusters - n_fixed_centroids
    
    centers = fixed_centroids
    
    if n_new_centroids < 0:
        raise "Yeah this is a problem because the algorithm won't do anything"
        
    pos = euclidean_distances(X = data, Y = centers).min(axis = 1)
    weighted_pos = [data_weights].transpose() * pos
    cumulative_pos = weighted_pos.sum()
    
    probabilities = np.full(n_points, 0, dtype = float)
    for i in range(0,n_points):
            probabilities[i] = weighted_pos[i]/cumulative_pos
    
    
    for j in range(n_fixed_centroids, n_clusters):
        sample_indices = choice(range(0,n_points), 2 + int(np.ln(n_clusters)), p = probabilities)
        samples = data[sample_indices]
        
        total_pot = euclidean_distances(samples, fixed_centroids).sum(axis = 1)
        top_candidate = np.argmax(total_pot) #slapdash but I think correct
        
        centers[j] = samples[top_candidate]
        
        pos = euclidean_distances(X = data, Y = centers).min(axis = 1)
        weighted_pos = data_weights * pos
        cumulative_pos = weighted_pos.sum()
    
        probabilities = np.full(n_points, 0, dtype = float)
        for i in range(0,n_points):
            probabilities[i] = weighted_pos[i]/cumulative_pos
    
    movable_centroids = centers[n_fixed_centroids-1:n_clusters-1]
    
    return centers, fixed_centroids, movable_centroids

def exploratory_kmeans(data, 
                       data_weighting, 
                       centers, 
                       fixed_centroids, 
                       movable_centroids):
    """
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    data_weighting : TYPE
        DESCRIPTION.
    fixed_centroids : TYPE
        DESCRIPTION.
    movable_centroids : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    n_movable, _ = fixed_centroids.shape
    n_clusters, n_features = centers.shape
    
    distances = euclidean_distances(X = data, Y = centers,squared = True)
    old_clusters = distances.argmin(axis = 1)
    new_clusters = np.array()
    counter = 0
    
    while old_clusters != new_clusters:
        if counter == 0:
            break
        else:
            old_clusters = new_clusters
            new_clusters = np.array()
        
        for cluster in range(n_movable, n_clusters):
            indices = np.where(old_clusters == cluster)
            group = data[indices]
            group_weights = [data_weighting[indices]].transpose()
        
            group = group * group_weights
            new_center = group.sum()/group_weights.sum()
        
            centers[cluster,] = new_center
            
        new_distances = euclidean_distances(data, centers, squared = True)
        new_clusters = new_distances.argmin(axis = 1)
        
    return centers, old_clusters
        
    