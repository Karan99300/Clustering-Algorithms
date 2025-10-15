# 🧠 Clustering Algorithms

A comprehensive **survey and implementation** of various clustering algorithms across multiple paradigms — from classic methods like K-Means to advanced density-based, graph-based, and model-based approaches.  

This repository explores the **theoretical foundations, practical performance, and hyperparameter behavior** of each algorithm through experiments on standard benchmark datasets.

---

## 📘 Overview

This project provides:
- Implementations of diverse **clustering algorithms**.
- A **comparative study** of their strengths and limitations.
- **Hyperparameter tuning** experiments to understand the impact of key parameters.
- **Visualizations** to intuitively explain clustering structures and results.

---

## ⚙️ Algorithm Categories

The repository covers clustering algorithms under multiple families:

| Category | Algorithms Included |
|-----------|--------------------|
| **Square-Error Based** | K-Means, Multiview K-Means |
| **Density-Based** | DBSCAN, DENCLUE, OPTICS |
| **Hierarchical** | Agglomerative Clustering, BIRCH |
| **Graph-Based** | Unnormalized Spectral Clustering, Zahn’s Clustering, DTG-Based Clustering |
| **Model-Based** | Expectation-Maximization (EM), Self-Organizing Maps (SOM) |
| **Search-Based / Swarm Intelligence** | Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Crow Search Optimization (CSO), Whale Optimization Algorithm (WOA) |
| **Autoencoder** | Autoencoder |

---

## 🔍 Hyperparameter Insights

Each algorithm was systematically tuned to evaluate the effects of its hyperparameters on clustering performance.  
Key evaluation metrics include:
- **Silhouette Score**
- **Calinski-Harabasz Index**
- **Davies-Bouldin Index**

Through this, the study identifies **parameter sensitivity**, **optimal configurations**, and **algorithm stability** across datasets.

---

## 📊 Datasets Used

Experiments were conducted on standard benchmark datasets:
- Breast Cancer
- Glass Identification
- Haberman’s Survival
- Iris
- Wine Recognition

---

## 🧩 Repository Structure

