import numpy as np
from scipy.spatial import Delaunay
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score

def calculate_delaunay_triangles(points):
    triangulation = Delaunay(points)
    return triangulation.simplices

def create_graph_from_delaunay_triangles(triangles):
    G = nx.Graph()
    for triangle in triangles:
        for i in range(3):
            for j in range(i+1, 3):
                G.add_edge(triangle[i], triangle[j])
    return G

def global_mean(graph):
    edge_lengths = [np.linalg.norm(np.array(graph.nodes[edge[0]]['pos']) - np.array(graph.nodes[edge[1]]['pos']))
                    for edge in graph.edges()]
    return np.mean(edge_lengths)

def mean_k(graph, vertex, k):
    neighbors = nx.single_source_shortest_path_length(graph, vertex, cutoff=k)
    edge_lengths = [np.linalg.norm(np.array(graph.nodes[edge[0]]['pos']) - np.array(graph.nodes[edge[1]]['pos']))
                    for edge in graph.edges() if edge[0] in neighbors and edge[1] in neighbors]
    return np.mean(edge_lengths)

def global_variation(graph, global_mean):
    edge_lengths = [
        np.linalg.norm(np.array(graph.nodes[edge[0]]['pos']) - np.array(graph.nodes[edge[1]]['pos']))
        for edge in graph.edges()
    ]
    N=len(edge_lengths)
    summation_result = sum(((ei - global_mean)**2 / (N - 1)) for ei in edge_lengths)
    global_variation_result = np.sqrt(summation_result)

    return global_variation_result
    
def global_cut_value(graph, vertex):
    global_mean_value = global_mean(graph)
    mean_1_value = mean_k(graph, vertex, 1)
    global_variation_value = global_variation(graph,global_mean_value)
    global_cut=global_mean_value + (global_mean_value / mean_1_value) * global_variation_value
    return global_cut

def remove_edges_above_cut_value(graph):
    edges_to_remove = []
    for edge in graph.edges():
        length = np.linalg.norm(np.array(graph.nodes[edge[0]]['pos']) - np.array(graph.nodes[edge[1]]['pos']))
        cut_value = global_cut_value(graph, edge[0])
        if length >= cut_value:
            edges_to_remove.append(edge)
    graph.remove_edges_from(edges_to_remove)
    return graph

def create_clusters(graph):
    return list(nx.connected_components(graph))

def local_variation(graph, vertex):
    neighbors = list(graph.neighbors(vertex))
    edge_lengths = [np.linalg.norm(np.array(graph.nodes[neighbor]['pos']) - np.array(graph.nodes[vertex]['pos']))
                    for neighbor in neighbors]
    n=len(edge_lengths)
    mean_1 = mean_k(graph, vertex, 1)
    summation_result = sum(((ei - mean_1) ** 2 / (n - 1)) for ei in edge_lengths)
    local_variation_result = np.sqrt(summation_result)

    return local_variation_result

def mean_variation(graph):
    local_variations = [local_variation(graph, vertex) for vertex in graph.nodes()]
    return np.mean(local_variations)

def local_cut_value(graph, vertex, beta):
    mean_k_value = mean_k(graph, vertex, 2)  
    mean_variation_value = mean_variation(graph)
    local_cut_value=mean_k_value + beta * mean_variation_value
    return local_cut_value

def remove_edges_above_local_cut_value(graph, beta=0.5):
    edges_to_remove = []
    for vertex in graph.nodes():
        local_cut_value_vertex = local_cut_value(graph, vertex, beta)
        first_order_neighbors = set(graph.neighbors(vertex))
        second_order_neighbors = set()
        for first_order_neighbor in first_order_neighbors:
            second_order_neighbors.update(graph.neighbors(first_order_neighbor))
        second_order_neighbors -= first_order_neighbors
        second_order_neighbors.discard(vertex)
        for neighbor in second_order_neighbors:
            length = np.linalg.norm(np.array(graph.nodes[neighbor]['pos']) - np.array(graph.nodes[vertex]['pos']))
            if length >= local_cut_value_vertex:
                edges_to_remove.append((vertex, neighbor))
    graph.remove_edges_from(edges_to_remove)
    return graph

def create_unit_vector_between_vertices(vertex1, vertex2):
    vector = vertex2 - vertex1
    magnitude = np.linalg.norm(vector)
    
    if magnitude == 0:
        raise ValueError("Cannot create a unit vector from a zero vector.")
    
    unit_vector = vector / magnitude
    return unit_vector
    
def angle_between_vectors(vector1,vector2):
    dot_product=np.dot(vector1,vector2)
    norm_vector1=np.linalg.norm(vector1)
    norm_vector2=np.linalg.norm(vector2)
    cosine_angle=dot_product/(norm_vector1*norm_vector2)
    angle=np.arccos(np.clip(cosine_angle,-1.0,1.0))
    return np.degrees(angle)

def local_aggregation_force(graph, source, target,k=1):
    pos_source = np.array(graph.nodes[source]['pos'])
    pos_target = np.array(graph.nodes[target]['pos'])
    distance = np.linalg.norm(pos_target - pos_source)
    force_direction = create_unit_vector_between_vertices(pos_source,pos_target)
    force_magnitude = k / distance**2
    force = force_direction * force_magnitude
    return force

def cohesive_local_aggregation_force(graph, vertex):
    cohesive_force=0
    first_order_neighbors = set(graph.neighbors(vertex))
    second_order_neighbors = set()
    for first_order_neighbor in first_order_neighbors:
        second_order_neighbors.update(graph.neighbors(first_order_neighbor))
    all_neighbors = first_order_neighbors.union(second_order_neighbors)
    all_neighbors-={vertex}
    for neighbor in all_neighbors:
        cohesive_force += local_aggregation_force(graph, vertex, neighbor)
    return cohesive_force

def local_aggregation_set(graph, vertex):
    local_agg_set = set()
    cohesive_force_vertex = cohesive_local_aggregation_force(graph, vertex)
    
    for neighbor in graph.neighbors(vertex):
        force_neighbor = local_aggregation_force(graph, vertex, neighbor)
        angle=angle_between_vectors(cohesive_force_vertex,force_neighbor)
        
        if angle < 90 and neighbor in graph.neighbors(vertex):
            local_agg_set.add(neighbor)
    
    return local_agg_set

def retain_edges_in_local_aggregation_set(graph):
    edges_to_remove = []
    for edge in graph.edges():
        if edge[1] not in local_aggregation_set(graph, edge[0]):
            edges_to_remove.append(edge)
    graph.remove_edges_from(edges_to_remove)
    return graph
    
def DTClustering(points,beta=1):
    triangles = calculate_delaunay_triangles(points)
    delaunay_graph = create_graph_from_delaunay_triangles(triangles)
    pos = {i: tuple(points[i]) for i in range(len(points))}
    nx.set_node_attributes(delaunay_graph, pos, 'pos')
    delaunay_graph=remove_edges_above_cut_value(delaunay_graph)
    subgraphs=list(nx.connected_components(delaunay_graph))
    for i,subgraph_nodes in enumerate(subgraphs):
        subgraph=delaunay_graph.subgraph(subgraph_nodes).copy()
        subgraph=remove_edges_above_local_cut_value(subgraph,beta=beta)

    subgraphs=list(nx.connected_components(delaunay_graph))
    for i,subgraph_nodes in enumerate(subgraphs):
        subgraph=delaunay_graph.subgraph(subgraph_nodes).copy()
        subgraph=retain_edges_in_local_aggregation_set(subgraph)
    clusters = create_clusters(delaunay_graph)
    return clusters
    
def visualize(points,clusters):
    for cluster_id, cluster in enumerate(clusters):
        cluster_points = points[list(cluster)]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id + 1}')
        
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    plt.figure(figsize=(10, 6))
    print(len(clusters))

    
def main():
        #Cluster 1   
    cluster1_num_samples = 20
    cluster1_x1_start = 0
    cluster1_x1_end = 5
    cluster1_x2_start = 2
    cluster1_x2_end = 6
    cluster1_x1 = np.random.random(size=(cluster1_num_samples))
    cluster1_x1 = cluster1_x1 * (cluster1_x1_end - cluster1_x1_start) + cluster1_x1_start
    cluster1_x2 = np.random.random(size=(cluster1_num_samples))
    cluster1_x2 = cluster1_x2 * (cluster1_x2_end - cluster1_x2_start) + cluster1_x2_start

    # Cluster 2
    cluster2_num_samples = 20
    cluster2_x1_start = 4
    cluster2_x1_end = 12
    cluster2_x2_start = 14
    cluster2_x2_end = 18
    cluster2_x1 = np.random.random(size=(cluster2_num_samples))
    cluster2_x1 = cluster2_x1 * (cluster2_x1_end - cluster2_x1_start) + cluster2_x1_start
    cluster2_x2 = np.random.random(size=(cluster2_num_samples))
    cluster2_x2 = cluster2_x2 * (cluster2_x2_end - cluster2_x2_start) + cluster2_x2_start

    # Cluster 3
    cluster3_num_samples = 20
    cluster3_x1_start = 12
    cluster3_x1_end = 18
    cluster3_x2_start = 8
    cluster3_x2_end = 11
    cluster3_x1 = np.random.random(size=(cluster3_num_samples))
    cluster3_x1 = cluster3_x1 * (cluster3_x1_end - cluster3_x1_start) + cluster3_x1_start
    cluster3_x2 = np.random.random(size=(cluster3_num_samples))
    cluster3_x2 = cluster3_x2 * (cluster3_x2_end - cluster3_x2_start) + cluster3_x2_start

    # Combine clusters into a single dataset
    c1 = np.array([cluster1_x1, cluster1_x2]).T
    c2 = np.array([cluster2_x1, cluster2_x2]).T
    c3 = np.array([cluster3_x1, cluster3_x2]).T

    data = np.concatenate((c1, c2, c3), axis=0)

    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=['X1', 'X2'])
    points=df[['X1','X2']].values
    clusters=DTClustering(points,beta=1)
    visualize(points,clusters)

if __name__ == '__main__':
    main()