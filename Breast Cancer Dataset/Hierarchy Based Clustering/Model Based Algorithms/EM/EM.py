import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def em(n_clusters, df):
  # Apply Gaussian Mixture Model
  gmm = GaussianMixture(n_components=n_clusters, random_state=42)
  cluster_labels = gmm.fit_predict(df)
  df['Cluster'] = cluster_labels
  return cluster_labels