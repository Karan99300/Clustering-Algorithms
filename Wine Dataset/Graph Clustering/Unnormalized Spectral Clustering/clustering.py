from scipy import sparse,linalg
from sklearn.neighbors import kneighbors_graph
from data import concentric_circle_generation
import numpy as np
from itertools import chain
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#Step1
def laplacian(df,nn):
    connectivity=kneighbors_graph(X=df,n_neighbors=nn,mode='connectivity')
    adj=(1/2)*(connectivity+connectivity.T)
    graph_laplacian=sparse.csgraph.laplacian(adj)
    graph_laplacian = graph_laplacian.toarray()
    return graph_laplacian
#Step2

def eigen(graph_laplacian):
    eigenvalues,eigenvectors=linalg.eig(graph_laplacian)
    eigenvalues=np.real(eigenvalues)
    eigenvectors=np.real(eigenvectors)
    return eigenvalues,eigenvectors

#Step3

def selecting_eigenvectors(eigenvalues,eigenvectors,n_clusters):
    eigenvalues=np.argsort(eigenvalues)
    indices=eigenvalues[:n_clusters]
    
    proj=pd.DataFrame(eigenvectors[:,indices.squeeze()])
    proj.columns=['v_' + str(c) for c in proj.columns]
    return proj

#Step 4

def k_means(df,n_clusters):
    kmeans=KMeans(random_state=42,n_clusters=n_clusters,n_init=10)
    kmeans.fit(df)
    cluster=kmeans.predict(df)
    return cluster

def spectral_clustering(df,n_neighbors,n_clusters):
    graph_laplacian=laplacian(df=df,nn=n_neighbors)
    eigenvals,eigenvects=eigen(graph_laplacian=graph_laplacian)
    proj=selecting_eigenvectors(eigenvalues=eigenvals,eigenvectors=eigenvects,n_clusters=n_clusters)
    return k_means(df=proj,n_clusters=n_clusters)
    
# Visualize clusters
def plot_clusters(df, cluster_labels):
    plt.scatter(df['X1'], df['X2'], c=cluster_labels, cmap='viridis', edgecolors='k')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Zahn\'s Clustering')
    plt.show()
    
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
    
    df['cluster'] = spectral_clustering(df, 5,3)
    plot_clusters(df, df['cluster'])
    silhouette_avg = silhouette_score(df[['X1', 'X2']], df['cluster'])
    print(f"Silhouette Score: {silhouette_avg}")

if __name__ == '__main__':
    main()
    