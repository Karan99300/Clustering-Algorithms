import numpy as np
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(3000)  
def distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Sample objective function
def objective_function(sol, centroids):
    total_distance = 0
    for i, cluster in enumerate(sol):
        centroid = centroids[i]
        for point in cluster:
            total_distance += distance(point, centroid)
    return total_distance

# Function to construct an initial solution
def initial_solution(dataset, Ns):
    centroids = dataset[np.random.choice(len(dataset), Ns, replace=False)]
    
    # Assign each data point to the closest centroid
    clusters = [[] for _ in range(Ns)]
    for point in dataset:
        closest_centroid = min(centroids, key=lambda centroid: distance(point, centroid))
        cluster_index = np.where(centroids == closest_centroid)[0][0]
        clusters[cluster_index].append(point)
    
    return clusters, centroids

# Function to perform intensification for a single cluster
def intensification_single_cluster(sol, centroids, cluster_index, bestSol, bestObjVal, optSol, optObjVal, nonImpIter, maxNonImpIter, tabu_list):
    cluster_i = sol[cluster_index]
    centroid_i = centroids[cluster_index]
    
    # Create neighborhood N(i)
    neighborhood_i = [point for cluster in sol if not np.array_equal(cluster, cluster_i) for point in cluster]

    
    # Create candidate list CL(i)
    candidate_list_i = sorted(neighborhood_i, key=lambda point: distance(point, centroid_i))
    
    # Reassign data points to the newly selected centroid
    for point in candidate_list_i:
        # Reassign data points based on your clustering requirements
        cluster_indices = np.where(np.all(centroids == centroid_i, axis=1))[0]
        
        if len(cluster_indices) > 0:
            cluster_indices = cluster_indices[0]
            sol[cluster_indices] = [p for p in sol[cluster_indices] if not np.array_equal(p, point)]
            sol[cluster_index].append(point)
            
            # Update centroids
            centroids[cluster_index] = np.mean(sol[cluster_index], axis=0)
            centroids[cluster_indices] = np.mean(sol[cluster_indices], axis=0)
    
    # Calculate the objective function value
    objVal = objective_function(sol, centroids)
    
    # Check if a new best solution is found
    if objVal < optObjVal:
        optObjVal = objVal
        optSol = sol
        nonImpIter = 0
    elif objVal < bestObjVal:
        bestObjVal = objVal
        bestSol = sol
        nonImpIter += 1

    # Record sol in the tabu list
    tabu_list.append(sol.copy())
    if len(tabu_list) > maxNonImpIter:
        tabu_list.pop(0)

    # Step 2.3: Repeat for all N(i)
    for i in range(len(sol)):
        if sol[i] not in tabu_list:
            intensification_single_cluster(sol, centroids, i, bestSol, bestObjVal, optSol, optObjVal, nonImpIter, maxNonImpIter, tabu_list)

# Function to perform intensification
def intensification(sol, centroids, bestSol, bestObjVal, optSol, optObjVal, nonImpIter, maxNonImpIter, tabu_list):
    # Repeat for all clusters
    for i in range(len(sol)):
        if sol[i] not in tabu_list:
            intensification_single_cluster(sol, centroids, i, bestSol, bestObjVal, optSol, optObjVal, nonImpIter, maxNonImpIter, tabu_list)

# Function to perform diversification
# Function to perform diversification
def diversification(sol, centroids, optSol, optObjVal, nonImpIter, maxDivIter, tabu_list):
    Ns = len(sol)
    
    # Repeat for a specified number of iterations
    for _ in range(maxDivIter):
        # Create neighborhoods for all clusters
        neighborhoods = [point for cluster in sol for point in cluster]
        
        # Perform diversification for each cluster based on the candidates created in Step 2.4
        for i in range(Ns):
            cluster_i = sol[i]
            centroid_i = centroids[i]

            # Create neighborhood N(i)
            neighborhood_i = [point for cluster in sol if cluster != cluster_i for point in cluster]
            
            # Create candidate list CL(i)
            candidate_list_i = sorted(neighborhood_i, key=lambda point: distance(point, centroid_i))
            
            # Reassign data points to the newly selected centroid
            for point in candidate_list_i:
                # Reassign data points based on your clustering requirements
                sol[np.where(centroids == centroid_i)[0][0]].remove(point)
                sol[i].append(point)
                
                # Update centroids
                centroids[i] = np.mean(sol[i], axis=0)
                centroids[np.where(centroids == centroid_i)[0][0]] = np.mean(sol[np.where(centroids == centroid_i)[0][0]], axis=0)
            
            # Calculate the objective function value
            objVal = objective_function(sol, centroids)
            
            # Check if a new best solution is found
            if objVal < optObjVal:
                optObjVal = objVal
                optSol = sol

        nonImpIter = 0  # Reset nonImpIter for intensification

# Main Tabu Search clustering algorithm
def tabu_search_clustering(dataset, Ns, maxNonImpIter, maxDivIter,tabu_tenure):
    # Step 1: Initial Solution Construction
    sol, centroids = initial_solution(dataset, Ns)
    optSol = bestSol = sol
    bestObjVal = optObjVal = objVal = objective_function(sol, centroids)

    nonImpIter = 0
    tabu_list = []

    # Step 2: Main Portion of the Tabu Search Algorithm
    while True:
        if nonImpIter < maxNonImpIter:
            # Intensification
            intensification(sol, centroids, bestSol, bestObjVal, optSol, optObjVal, nonImpIter, maxNonImpIter, tabu_list)
        else:
            # Diversification
            diversification(sol, centroids, optSol, optObjVal, nonImpIter, maxDivIter, tabu_list)
            nonImpIter = 0  # Reset nonImpIter for intensification

        # Termination condition (you can customize this based on your criteria)
        objVal = objective_function(sol, centroids)

        # Check if the stopping condition is met
        if stopping_condition(objVal, prevObjVal):
            break
        
        tabu_list.append(sol.copy())

        # Check tabu tenure expiry
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        prevObjVal = objVal

    # Step 3: Output the result
    return optSol, optObjVal

# Example stopping condition based on the change in objective function
def stopping_condition(obj_val, prev_obj_val, threshold=0.01):
    return abs(obj_val - prev_obj_val) / prev_obj_val < threshold

# Function to visualize clusters and centroids
def visualize_clusters(sol, centroids):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, cluster in enumerate(sol):
        points = np.array(cluster)
        centroid = centroids[i]
        
        # Plot data points in the cluster
        plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i + 1}', alpha=0.7)
        
        # Plot centroid
        plt.scatter(centroid[0], centroid[1], marker='x', s=200, c=colors[i], label=f'Centroid {i + 1}')

    plt.title('Cluster Visualization')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def main():
    cluster1_num_samples = 20
    cluster1_x1_start = 0
    cluster1_x1_end = 5
    cluster1_x2_start = 2
    cluster1_x2_end = 6
    cluster1_x1 = np.random.random(size=(cluster1_num_samples))
    cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
    cluster1_x2 = np.random.random(size=(cluster1_num_samples))
    cluster1_x2 = cluster1_x2 * (cluster1_x2_end - cluster1_x2_start) + cluster1_x2_start

    cluster2_num_samples = 20
    cluster2_x1_start = 4
    cluster2_x1_end = 12
    cluster2_x2_start = 14
    cluster2_x2_end = 18
    cluster2_x1 = np.random.random(size=(cluster2_num_samples))
    cluster2_x1 = cluster2_x1 * (cluster2_x1_end - cluster2_x1_start) + cluster2_x1_start
    cluster2_x2 = np.random.random(size=(cluster2_num_samples))
    cluster2_x2 = cluster2_x2 * (cluster2_x2_end - cluster2_x2_start) + cluster2_x2_start

    cluster3_num_samples = 20
    cluster3_x1_start = 12
    cluster3_x1_end = 18
    cluster3_x2_start = 8
    cluster3_x2_end = 11
    cluster3_x1 = np.random.random(size=(cluster3_num_samples))
    cluster3_x1 = cluster3_x1 * (cluster3_x1_end - cluster3_x1_start) + cluster3_x1_start
    cluster3_x2 = np.random.random(size=(cluster3_num_samples))
    cluster3_x2 = cluster3_x2 * (cluster3_x2_end - cluster3_x2_start) + cluster3_x2_start

    c1 = np.array([cluster1_x1, cluster1_x2]).T
    c2 = np.array([cluster2_x1, cluster2_x2]).T
    c3 = np.array([cluster3_x1, cluster3_x2]).T

    data = np.concatenate((c1, c2, c3), axis=0)
    
    Ns = 3
    maxNonImpIter = 10
    maxDivIter = 5
    result_sol, result_obj_val = tabu_search_clustering(data, Ns, maxNonImpIter, maxDivIter,tabu_tenure=5)
    result_centroids = [np.mean(cluster, axis=0) for cluster in result_sol]
    visualize_clusters(result_sol, result_centroids)
    
if __name__ == '__main__':
    main()
