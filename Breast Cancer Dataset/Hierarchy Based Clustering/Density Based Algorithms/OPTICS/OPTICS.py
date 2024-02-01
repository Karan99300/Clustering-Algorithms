import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt

def optics(D, MinPts, eps):
    # Initialize unprocessed points
    unprocessed = set(range(len(D)))
    
    # Initialize priority queue PQ
    PQ = PriorityQueue()
    
    # Initialize a list to store reachability distances
    reachability_distances = [float('inf')] * len(D)
    
    def calculate_core_distance(P, MinPts):
        # Calculate distance to MinPts-th nearest neighbor of P
        distances = [np.linalg.norm(D[P] - D[other]) for other in range(len(D)) if other != P]
        distances.sort()
        return distances[MinPts - 1]
    
    def update_neighbors(P, core_dist, MinPts, PQ, eps):
        nonlocal unprocessed
        
        for Q in unprocessed.copy():
            if Q != P:
                distance_P_Q = np.linalg.norm(D[P] - D[Q])
                reach_dist = max(core_dist, distance_P_Q)
                
                if reach_dist <= eps:
                    PQ.put((reach_dist, Q))
                    unprocessed.remove(Q)
                    reachability_distances[Q] = reach_dist
    
    # Main OPTICS algorithm
    for P in range(len(D)):
        if P in unprocessed:
            core_dist = calculate_core_distance(P, MinPts)
            unprocessed.remove(P)
            update_neighbors(P, core_dist, MinPts, PQ, eps)
    
    # Extract cluster labels
    cluster_labels = extract_clusters(reachability_distances, eps)
    
    # # Plot reachability distances
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(D)), reachability_distances, marker='o', linestyle='-', color='b')
    # plt.xlabel('Points')
    # plt.ylabel('Reachability Distance')
    # plt.title('Reachability Plot')
    # plt.show()

    return cluster_labels

def extract_clusters(reachability_distances, eps):
    # Initialize a list to store cluster labels
    cluster_labels = [-1] * len(reachability_distances)

    current_cluster = 0

    for i, reach_dist in enumerate(reachability_distances):
        if reach_dist > eps:
            if cluster_labels[i] == -1:
                current_cluster += 1
                cluster_labels[i] = current_cluster

            for j in range(i + 1, len(reachability_distances)):
                if reachability_distances[j] <= eps:
                    cluster_labels[j] = current_cluster
                else:
                    break

    return cluster_labels