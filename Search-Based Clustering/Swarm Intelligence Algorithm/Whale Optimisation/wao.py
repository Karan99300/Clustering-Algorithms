import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def initialize_search_agents(num_agents, num_clusters, data):
    agents = []
    for _ in range(num_agents):
        random_centers = data[np.random.choice(len(data), num_clusters, replace=False)]
        agents.append({'cluster_centers': random_centers})
    
    return agents

def assign_to_clusters(agent, data):
    clusters = [[] for _ in range(len(agent['cluster_centers']))]
    for point in data:
        distances = [euclidean_distance(point, center) for center in agent['cluster_centers']]
        closest_cluster = np.argmin(distances)
        clusters[closest_cluster].append(point)
    return clusters

def calculate_fitness(agent, data):
    total_fitness = 0
    clusters = assign_to_clusters(agent, data)
    for i, cluster_center in enumerate(agent['cluster_centers']):
        for point in clusters[i]:
            total_fitness += euclidean_distance(point, cluster_center)  
    return total_fitness

def update_search_agent(agents,agent, best_agent, a, p):
    A = 2 * a * np.random.rand() - a
    C = 2 * np.random.rand()
    
    if p < 0.5:
        if np.abs(A) < 1:
            D = np.abs(C * best_agent['cluster_centers'] - agent['cluster_centers'])
            agent['cluster_centers'] = best_agent['cluster_centers'] - A * D
        else:
            random_agent = np.random.choice(agents)
            D = np.abs(C * random_agent['cluster_centers'] - agent['cluster_centers'])
            agent['cluster_centers'] = random_agent['cluster_centers'] - A * D
    else:
        b = 0.1
        l = np.random.uniform(-1, 1, size=(len(agent['cluster_centers']),))  
        D_prime = np.abs(best_agent['cluster_centers'] - agent['cluster_centers'])
        agent['cluster_centers'] = D_prime * np.exp(b*l[:, np.newaxis]) * np.cos(2*np.pi*l[:, np.newaxis]) + best_agent['cluster_centers']


def whale_optimization_algorithm(data, num_clusters, num_agents, iterations):
    agents = initialize_search_agents(num_agents, num_clusters, data)

    for t in range(iterations):
        best_agent = min(agents, key=lambda agent: calculate_fitness(agent, data))
        
        for agent in agents:
            a = 2 - t * (2 / iterations) 
            p = np.random.rand()
            
            update_search_agent(agents,agent, best_agent, a, p)

    return best_agent

def visualize_clusters(data, cluster_centers, assignments):
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', marker='o', edgecolors='k')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
    plt.title('Whale Optimisation Algorithm - Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
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

    num_clusters=3
    num_agents=100
    num_iterations=1000
    best_agent=whale_optimization_algorithm(data,num_clusters,num_agents,num_iterations)
    assignments = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - best_agent['cluster_centers'], axis=2), axis=1)
    visualize_clusters(data,best_agent['cluster_centers'],assignments)
    
if __name__ == '__main__':
    main()