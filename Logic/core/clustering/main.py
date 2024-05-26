########## needed for relative import ##########
import inspect
import sys
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
################################################

import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb

from word_embedding.fasttext_data_loader import FastTextDataLoader
from word_embedding.fasttext_model import FastText
from word_embedding.fasttext_model import preprocess_text
from classification.data_loader import ReviewLoader
try:
    from .dimension_reduction import DimensionReduction
    from .clustering_metrics import ClusteringMetrics
    from .clustering_utils import ClusteringUtils
except ImportError:
    from dimension_reduction import DimensionReduction
    from clustering_metrics import ClusteringMetrics
    from clustering_utils import ClusteringUtils


LOAD_EMBEDDINGS = True

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.

# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.

# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.

# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.

def main():
    # Assuming movie_embeddings and true_movie_labels are defined elsewhere
    # movie_embeddings =...
    # true_movie_labels =...

    # Initialize WandB
    # wandb.init(project="MIR-2024-Project", name="Movie Clustering")

    # 0.
    fasttext_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'word_embedding', 'FastText_model.bin')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utility', 'IMDB_crawled.json')
    ft_data_loader = FastTextDataLoader(path)
    X, y, _ = ft_data_loader.create_train_data()
    fasttext_model = FastText(preprocessor=preprocess_text)
    fasttext_model.prepare(None, mode = "load", path=fasttext_model_path)
    if LOAD_EMBEDDINGS:
        with open('embeddings.npy', 'rb') as f:
            embeddings = np.load(f)
    else:
        embeddings = np.array(list(map(fasttext_model.get_query_embedding, X)))
        with open('embeddings.npy', 'wb') as f:
            np.save(f, embeddings)

    n_components = 2
    # 1.
    dr = DimensionReduction()
    pca_feats = dr.pca_reduce_dimension(embeddings, n_components=n_components)
    dr.wandb_plot_explained_variance_by_components()
    tsne_feats = dr.convert_to_2d_tsne(emb_vecs=embeddings[:100, :])
    dr.wandb_plot_2d_tsne(tsne_feats)


    # 2.
    clustering_utils = ClusteringUtils()
    cm = ClusteringMetrics()
    clustering_utils.plot_kmeans_cluster_scores(embeddings, y, [3, 5, 7, 9, 11], project_name="MIR-2024-Project", run_name="Movie Clustering")
    # 7 seems to be a good value
    centroids, cluster_indices = clustering_utils.cluster_kmeans(embeddings, 9)
    print(f'purity value for 7 clusters: {cm.purity_score(y, cluster_indices)}')

    # 3.
    linkage_methods = ['single', 'complete', 'average', 'ward']
    for linkage_method in linkage_methods:
        cluster_indices = clustering_utils.cluster_hierarchical(emb_vecs=embeddings, linkage_method=linkage_method)
        # Convert y and cluster_indices to lists of strings
        y_str = list(map(str, y))
        cluster_indices_str = list(map(str, cluster_indices))
        print(f'Purity value for {linkage_method} linkage: {cm.purity_score(y_str, cluster_indices_str)}')
        clustering_utils.visualize_hierarchical_clustering_dendrogram(data=embeddings,
         linkage_method=linkage_method, project_name="MIR-2024-Project", run_name=f"Hierarchical_{linkage_method}")

        # wandb.log({f"{linkage_method}_dendrogram": dendrogram_plot})
        # wandb.log({f"{linkage_method}_cluster_assignments": cluster_assignments_plot})
    exit()
    wandb.finish()

if __name__ == "__main__":
    main()
