import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
# Visualize clusters using PCA for dimensionality reduction

def visualization(x, cluster_labels):
  pca = PCA(n_components=3)  # Use 3 principal components for 3D visualization
  X_pca = pca.fit_transform(x)

  # Create a 3D scatter plot
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='viridis', alpha=0.5)
  ax.set_title('DBSCAN Clustering 3D Visualization')
  ax.set_xlabel('Principal Component 1')
  ax.set_ylabel('Principal Component 2')
  ax.set_zlabel('Principal Component 3')

  plt.show()


def interactive_visualization(x, cluster_labels):
  pca = PCA(n_components=3)  # Use 3 principal components for 3D visualization
  X_pca = pca.fit_transform(x)
  
  fig = go.Figure()

  fig.add_trace(go.Scatter3d(
      x=X_pca[:, 0],
      y=X_pca[:, 1],
      z=X_pca[:, 2],
      mode='markers',
      marker=dict(color=cluster_labels, colorscale='viridis', opacity=0.5),
      text='Cluster Labels'
  ))

  fig.update_layout(scene=dict(
                      xaxis_title='Principal Component 1',
                      yaxis_title='Principal Component 2',
                      zaxis_title='Principal Component 3'),
                    title='DBSCAN Clustering 3D Visualization',
                    margin=dict(l=0, r=0, b=0, t=0))

  fig.show()



