import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

def initialize_solution(dataset, Ns):
    centroids_indices = np.random.choice(len(dataset), size=Ns, replace=False)
    centroids = dataset[centroids_indices]

    sol = [list() for _ in range(Ns)]
    for point in dataset:
        closest_centroid_idx = np.argmin([euclidean_distance(point, centroid) for centroid in centroids])
        sol[closest_centroid_idx].append(point)

    return sol,centroids

def objective_function(sol, centroids):
    total_distance = sum(np.sum([euclidean_distance(point, centroid) for point in cluster]) for cluster, centroid in zip(sol, centroids))
    return total_distance

def stopping_criteria(obj_values, stabilization_iter=5, threshold=0.01):
    if len(obj_values) >= stabilization_iter:
        recent_obj_values = obj_values[-stabilization_iter:]
        avg_change = np.mean(np.abs(np.diff(recent_obj_values)))

        if avg_change < threshold:
            return True

    return False

def calculate_radius(cluster, centroid):
    if len(cluster) > 0:
        total_distance = sum(euclidean_distance(point, centroid) for point in cluster)
        radius = total_distance / len(cluster)
        return radius
    else:
        # Return a default radius or handle the case appropriately
        return 0.0  # You may adjust this value based on your specific requirements


def create_neighborhood(cluster, radius_multiplier=1.0):
    current_centroid = np.mean(cluster, axis=0)
    radius = calculate_radius(cluster, current_centroid) * radius_multiplier
    neighborhood_i = [point for point in cluster if euclidean_distance(point, current_centroid) <= radius]
    
    return neighborhood_i

def calculate_total_distance_to_other_points(centroid, cluster, point_to_exclude):
    total_distance = sum(euclidean_distance(point, centroid) for point in cluster if not np.array_equal(point, point_to_exclude))
    return total_distance

def create_candidate_list(cluster, neighborhood, prev_optimal_distance):
    centroid = np.mean(cluster, axis=0)
    v_values = [calculate_total_distance_to_other_points(centroid, cluster, point) for point in neighborhood]
    
    # Step 1: Sort data points in N(i) in ascending order of their V(j) values
    sorted_indices = np.argsort(v_values)
    
    # Step 2: Determine epsilon
    epsilon = sum(v < prev_optimal_distance for v in v_values)
    
    # Step 3: Determine lambda
    high_value = max(1, len(neighborhood) - epsilon)
    lambda_value = np.random.randint(0, high_value)
    
    # Step 4: Set n(i) = epsilon + lambda
    n_i = epsilon + lambda_value
    
    # Ensure n(i) is within the valid range
    n_i = min(len(sorted_indices), max(0, n_i))
    
    # Create candidate list CL(i)
    candidate_list = [neighborhood[sorted_indices[i]] for i in range(n_i)]
    print("epsilon:", epsilon)
    print("lambda_value:", lambda_value)
    print("n_i:", n_i)
    
    return candidate_list




def intensification_single_cluster(sol, centroids, cluster_index, optSol, bestSol, optObjVal, bestObjVal, objVal, tabu_list, nonImpIter, tabu_tenure):
    cluster = sol[cluster_index]
    neighborhood_i = create_neighborhood(cluster)
    candidate_list_i = create_candidate_list(cluster, neighborhood_i, optObjVal)

    for point in candidate_list_i:
        new_centroid = np.mean([p for p in cluster if not np.array_equal(p, point)], axis=0)
        new_cluster = [p for p in neighborhood_i if euclidean_distance(p, new_centroid) <= calculate_radius(cluster, new_centroid)]

        move = (tuple(cluster), tuple(new_cluster))
        if not any(np.array_equal(move, tabu_move) for tabu_move, _ in tabu_list):
            sol[cluster_index] = new_cluster
            objVal = objective_function(sol, centroids)

            if objVal < optObjVal:
                optObjVal = bestObjVal = objVal
                optSol = bestSol = sol.copy()
                nonImpIter = 0
            elif objVal < bestObjVal:
                bestSol = sol.copy()
                bestObjVal = objVal
                nonImpIter += 1

            # Record move in tabu list
            tabu_list.append((move, tabu_tenure))

            break

    tabu_list = [(tabu_move, tenure - 1) for tabu_move, tenure in tabu_list if tenure > 0]
    return optSol, optObjVal, nonImpIter


def intensification_all_clusters(sol, centroids, optSol, bestSol, optObjVal, bestObjVal, objVal, tabu_list, nonImpIter, tabu_tenure):
    for i in range(len(sol)):
        optSol, optObjVal, nonImpIter = intensification_single_cluster(sol, centroids, i, optSol, bestSol, optObjVal, bestObjVal, objVal, tabu_list, nonImpIter, tabu_tenure)
    return optSol, optObjVal, nonImpIter

def diversification(sol, centroids, optSol, optObjVal, bestSol, bestObjVal, objVal, tabu_list, nonImpIter, maxDivIter, tabu_tenure):
    for _ in range(maxDivIter):
        for i in range(len(sol)):
            neighborhood_i = create_neighborhood(sol[i])
            candidate_list_i = create_candidate_list(sol[i], neighborhood_i, optObjVal)

            for point in candidate_list_i:
                new_centroid = np.mean([p for p in sol[i] if not np.array_equal(p, point)], axis=0)
                new_cluster = [p for p in neighborhood_i if euclidean_distance(p, new_centroid) <= calculate_radius(sol[i], new_centroid)]

                move = (tuple(sol[i]), tuple(new_cluster))
                if move not in tabu_list:
                    sol[i] = new_cluster
                    objVal = objective_function(sol, centroids)

                    if objVal < optObjVal:
                        optObjVal = bestObjVal = objVal
                        optSol = bestSol = sol.copy()
                        nonImpIter = 0
                    elif objVal < bestObjVal:
                        bestSol = sol.copy()
                        bestObjVal = objVal
                        nonImpIter += 1

                    tabu_list.append((move, tabu_tenure))
                    break

        tabu_list = [(move, tenure - 1) for move, tenure in tabu_list if tenure > 0]
        nonImpIter = 0  # Reset nonImpIter for running intensification again

    return optSol, optObjVal,nonImpIter


def tabu_search_clustering(dataset, Ns, maxNonImpIter,maxDivIter,maxiterations,tabu_tenure):
    sol,centroids=initialize_solution(dataset,Ns)
    optSol=bestSol=sol.copy()
    
    objVal=objective_function(sol,centroids)
    optObjVal=bestObjVal=objVal
    nonImpIter=0
    tabu_list=[]
    
    obj_val=[objVal]
    
    for iteration in range(maxiterations):
        while not stopping_criteria(obj_val):
            if nonImpIter<maxNonImpIter:
                optSol,optObjVal,nonImpIter=intensification_all_clusters(sol,centroids,optSol,bestSol,optObjVal,bestObjVal,objVal,tabu_list,nonImpIter,tabu_tenure)
            else:
                optSol,optObjVal,nonImpIter=diversification(sol,centroids,optSol,bestSol,optObjVal,bestObjVal,objVal,tabu_list,maxDivIter,tabu_tenure)
            obj_val.append(optObjVal)
                
    return optSol,optObjVal

def visualize_clusters(sol, centroids):
    print("sol:", sol)
    print("centroids:", centroids)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, cluster in enumerate(sol):
        points = np.array(cluster)
        centroid = centroids[i]
        
        # Plot data points in the cluster
        plt.scatter(points, [0] * len(points), c=colors[i], label=f'Cluster {i + 1}', alpha=0.7)
        
        # Plot centroid
        plt.scatter(centroid, 0, marker='x', s=200, c=colors[i], label=f'Centroid {i + 1}')

    plt.title('Cluster Visualization')
    plt.xlabel('Feature 1')
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
    result_sol, result_obj_val = tabu_search_clustering(data, Ns, maxNonImpIter, maxDivIter,maxiterations=100,tabu_tenure=5)
    result_centroids = [np.mean(cluster, axis=0) for cluster in result_sol]
    visualize_clusters(result_sol, result_centroids)
    
if __name__ == '__main__':
    main()
    
    