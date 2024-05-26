import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
try:
    from .clustering_metrics import *
except ImportError:
    from clustering_metrics import *


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        num_samples, num_features = emb_vecs.shape
        initial_indices = random.sample(range(num_samples), n_clusters)
        centers = emb_vecs[initial_indices]
        labels = np.zeros(num_samples, dtype=int)

        for iteration in range(max_iter):
            for i in range(num_samples):
                distances = np.linalg.norm(emb_vecs[i] - centers, axis=1)
                labels[i] = np.argmin(distances)
            new_centers = np.array([emb_vecs[labels == k].mean(axis=0) for k in range(n_clusters)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        return centers.tolist(), labels.tolist()

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        all_words = [word for doc in documents for word in doc.split()]
        counter = Counter(all_words)
        most_common_words = counter.most_common(top_n)
        return [(word, freq) for word, freq in most_common_words]

    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        num_samples = len(emb_vecs)
        centroids = [emb_vecs[i] for i in random.sample(range(num_samples), n_clusters)]
    
        for _ in range(100):
            cluster_assignments = [[] for _ in range(n_clusters)]
        
            for vec in emb_vecs:
                distances = [np.linalg.norm(vec - centroid) for centroid in centroids]
                cluster_index = distances.index(min(distances))
                cluster_assignments[cluster_index].append(vec)
        
            new_centroids = [[np.mean([sample[i] for sample in cluster_assignment]) for i in range(len(sample))] for cluster_assignment in cluster_assignments]
        
            if np.allclose(new_centroids, centroids):
                break  
        
            centroids = new_centroids
    
        cluster_indices = [min(distances.index(np.linalg.norm(vec - centroid)) for centroid in centroids) for vec in emb_vecs]

        wcss = sum([sum([np.linalg.norm(vec - centroid)**2 for vec in cluster_assignments[i]]) for i, cluster_assignments in enumerate(cluster_assignments)])
    
        return centroids, cluster_indices, wcss

    def cluster_hierarchical(self, linkage_method: str, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with the specified linkage method.

        Parameters
        -----------
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of "single", "complete", "average", or "ward".
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        Z = linkage(emb_vecs, linkage_method)
        cluster_indices = fcluster(Z, t=0.5 * max(Z[:, 2]), criterion='maxclust')
        return cluster_indices.tolist()

    def visualize_hierarchical_clustering_dendrogram(self, data, linkage_method, project_name, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """

        Z = linkage(data, linkage_method)

        plt.figure(figsize=(10, 7))
        dn = dendrogram(Z)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method})')
        plt.xlabel("Data Points")
        plt.ylabel("Distance")

        # wandb.log({"Hierarchical Clustering Dendrogram": wandb.Image(plt.gcf())})

        plt.show()
        plt.close()

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        Z = linkage(emb_vecs, 'single')
        cluster_indices = fcluster(Z, t=0.5 * max(Z[:, 2]), criterion='maxclust')
        return cluster_indices.tolist()

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        Z = linkage(emb_vecs, 'complete')
        cluster_indices = fcluster(Z, t=0.5 * max(Z[:, 2]), criterion='maxclust')
        return cluster_indices.tolist()

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        Z = linkage(emb_vecs, 'average')
        cluster_indices = fcluster(Z, t=0.5 * max(Z[:, 2]), criterion='maxclust')
        return cluster_indices.tolist()

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        Z = linkage(emb_vecs, 'ward')
        cluster_indices = fcluster(Z, t=0.5 * max(Z[:, 2]), criterion='maxclust')
        return cluster_indices.tolist()

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """

        centroids, cluster_indices = self.cluster_kmeans(data, n_clusters)
    
        tsne_data = self.convert_to_2d_tsne(data)
    
        plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']
        for i in range(n_clusters):
            plt.scatter(tsne_data[cluster_indices == i, 0], tsne_data[cluster_indices == i, 1], c=colors[i % len(colors)], label=f'Cluster {i}')
    
        plt.title(f'K-Means Clustering with {n_clusters} Clusters')
        plt.legend()
    
        wandb.log({f'K-Means Clustering': wandb.Image(plt.gcf())})

        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """

        Z = linkage(data, linkage_method)

        plt.figure(figsize=(10, 7))
        dn = dendrogram(Z)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method})')
        plt.xlabel("Data Points")
        plt.ylabel("Distance")

        wandb.log({"Hierarchical Clustering Dendrogram": wandb.Image(plt.gcf())})

        plt.close()

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        cm = ClusteringMetrics()

        for k in k_values:
            centroids, cluster_indices = self.cluster_kmeans(embeddings, k)
            silhouette_scores.append(cm.silhouette_score(embeddings, cluster_indices))
            purity_scores.append(cm.purity_score(true_labels, cluster_indices))

        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, silhouette_scores, marker='o')
        plt.title('Silhouette Scores vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')

        plt.subplot(1, 2, 2)
        plt.plot(k_values, purity_scores, marker='o')
        plt.title('Purity Scores vs. Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Purity Score')

        # if project_name and run_name:
        #     run = wandb.init(project=project_name, name=run_name)
        #     wandb.log({"Cluster Scores": wandb.Image(plt.gcf())})

        plt.show()
        plt.close()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """

        wcss_values = []
        for k in k_values:
            centroids, cluster_indices, wcss = self.cluster_kmeans_WCSS(embeddings, k)
            wcss_values.append(wcss)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss_values, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.grid(True)

        wandb.log({"Elbow Method": wandb.Image(plt.gcf())})


        plt.close()