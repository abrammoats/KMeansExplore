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
    weighted_pos = data_weights.transpose() * pos
    cumulative_pos = weighted_pos.sum()
    
    probabilities = np.full(n_points, 0, dtype = float)
    for i in range(0,n_points):
        probabilities[i,] = weighted_pos[0][i,]/cumulative_pos
    
    wayward_sum = probabilities[np.where(probabilities < 0)].sum()
    probabilities[np.where(probabilities < 0)] = 0
    probabilities[0] = probabilities[0] + wayward_sum
    difference = float(1) - probabilities.sum()
    probabilities[0] = probabilities[0] + difference
    for j in range(n_fixed_centroids, n_clusters):
        sample_indices = choice(range(0,n_points), 2 + int(np.log(n_clusters)), p = probabilities)
        samples = data[sample_indices]
        
        total_pot = euclidean_distances(samples, fixed_centroids).sum(axis = 1)
        top_candidate = np.argmax(total_pot) #slapdash but I think correct
        
        centers = np.append(centers, [samples[top_candidate]], axis = 0)
        
        pos = euclidean_distances(X = data, Y = centers).min(axis = 1)
        weighted_pos = data_weights.transpose() * pos
        cumulative_pos = weighted_pos.sum()
    
        probabilities = np.full(n_points, 0, dtype = float)
        for i in range(0,n_points):
            probabilities[i,] = weighted_pos[0][i,]/cumulative_pos
            
        wayward_sum = probabilities[np.where(probabilities < 0)].sum()
        probabilities[np.where(probabilities < 0)] = 0
        probabilities[0] = probabilities[0] + wayward_sum
        difference = float(1) - probabilities.sum()
        probabilities[0] = probabilities[0] + difference
    
    movable_centroids = centers[n_fixed_centroids-1:n_clusters-1]
    
    return centers, fixed_centroids, movable_centroids

def single_exploratory_kmeans(data, 
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
    
    distances = euclidean_distances(X = data, Y = centers)
    old_clusters = np.argmin(distances, axis = 1)
    new_clusters = np.array([])
    counter = 0
    
    while not np.array_equal(old_clusters, new_clusters):
        if counter == 0:
            pass
        else:
            old_clusters = new_clusters
            new_clusters = np.array([])
        
        for cluster in range(n_movable, n_clusters):
            indices = np.where(old_clusters == cluster)
            group = data[indices]
            group_weights = data_weighting[indices].transpose()
        
            group = group * group_weights.transpose()
            new_center = group.sum(axis = 0)/group_weights.sum()
        
            centers[cluster,] = new_center
            
        new_distances = euclidean_distances(data, centers)
        new_clusters = new_distances.argmin(axis = 1)
        counter = counter + 1
        
    return centers, old_clusters


def expansion_kmeans(n_clusters, data, data_weights, fixed_centroids):
    """

    Parameters
    ----------
    n_clusters : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    data_weights : TYPE
        DESCRIPTION.
    fixed_centroids : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """     
    centers, fixed_centers, movable_centers = exploratory_kmeans_plusplus(n_clusters, 
                                                                          data, 
                                                                          data_weights, 
                                                                          fixed_centroids)
    
    champion_centers, _ = single_exploratory_kmeans(data, 
                       data_weights, 
                       centers, 
                       fixed_centers, 
                       movable_centers)
    champion_distances = euclidean_distances(data, champion_centers).min(axis = 1)
    champion_weighted_distances = champion_distances * data_weights.transpose()
    champion_weighted_error = champion_weighted_distances.sum()
    
    for i in range(0,300):
        print(i)
        challenger_centers,_ = single_exploratory_kmeans(data, 
                                                        data_weights, 
                                                        centers, 
                                                        fixed_centers, 
                                                        movable_centers)
        
        challenger_distances = euclidean_distances(data, challenger_centers).min(axis = 1)
        challenger_weighted_distances = challenger_distances * data_weights.transpose()
        challenger_weighted_error = challenger_weighted_distances.sum()
        
        if challenger_weighted_error < champion_weighted_error:
            champion_centers = challenger_centers.copy()
            
    return champion_centers
