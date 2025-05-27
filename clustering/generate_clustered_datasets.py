import torch
from openai import OpenAI
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from itertools import product


dataset_paths = ['assist2009/skill_builder_data_corrected_collapsed.csv', 
 'assist2012/2012-2013-data-with-predictions-4-final.csv', 
 'assist2017/anonymized_full_release_competition_dataset.csv']
datasets = ['assist2009', 'assist2012', 'assist2017']
dimensions = [2, 3, 4, 5, 6]
# dim_methods = ['tsne', 'umap', 'pca']
dim_methods = ['tsne', 'pca']
n_cluster_ranges = [range(38, 169, 10),range(92, 203, 10),range(28, 159, 10)]


for (dataset, dataset_path, n_cluster_range), n_component, dim_method in product(zip(datasets, dataset_paths, n_cluster_ranges), dimensions, dim_methods):
    problem_id_column_name = 'problem_id'
    if dataset == 'assist2017':
        problem_id_column_name = 'problemId'

    print(f"Processing dataset: {dataset} with {dim_method} and {n_component} components")
    emb_filenames = torch.load('embeddings/' + dataset + '/filenames.pt')
    pb_subset_df = pd.read_csv('data_subsets/' + dataset + '/questions.csv')

    for emb_filename in emb_filenames:
        # Extract filename without path and extension
        filename = os.path.basename(emb_filename)
        filename_base = os.path.splitext(filename)[0]

        # Create folder for the current file
        cluster_folder = os.path.join('clusters/' + dataset , filename_base)
        os.makedirs(cluster_folder, exist_ok=True)

        # Load the tensor
        skills_tensor = torch.load(emb_filename)

        # Convert tensor to numpy for scikit-learn
        skills_numpy = skills_tensor.numpy()
        # Perform dimensionality reduction based on dim_method
        if dim_method == 'tsne':
            reducer = TSNE(n_components=n_component, random_state=42)
        elif dim_method == 'umap':
            reducer = umap.UMAP(n_components=n_component, random_state=42)
        elif dim_method == 'pca':
            reducer = PCA(n_components=n_component, random_state=42)

        else:
            raise ValueError(f"Unknown dimensionality reduction method: {dim_method}")

        skills_tsne = reducer.fit_transform(skills_numpy)

        # Perform k-means clustering for different values of n
        for n_clusters in n_cluster_range:
            #################################    Mahdi added these parts   ###################
            cluster_output_path = os.path.join(cluster_folder, f"clusters_n{n_clusters}.pt")
            csv_output_path = os.path.join(cluster_folder, f"{dim_method}_{n_component}_n{n_clusters}.csv")

            if os.path.exists(cluster_output_path) and os.path.exists(csv_output_path):
                print(f"✓ {dataset}/{filename_base} – {n_clusters} clusters already done, skipping")
                continue
            
            ####################################################################################


            print(f"Clustering {filename_base} with {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Added n_init for newer sklearn versions
            clusters = kmeans.fit_predict(skills_tsne)

            # Save the clustering results
            cluster_output_path = os.path.join(cluster_folder, f'clusters_n{n_clusters}.pt')
            torch.save(torch.tensor(clusters), cluster_output_path)
            print(f"Saved clusters to {cluster_output_path}")
            csv_filename = os.path.basename(dataset_path)
            assist_df_pb_subset = pd.read_csv('data_subsets/' + dataset + '/' + csv_filename, encoding = "ISO-8859-1")
            pb_subset_df['meta-kc'] = clusters
            assist_df_pb_subset = assist_df_pb_subset.merge(pb_subset_df[[problem_id_column_name, 'meta-kc']], on=problem_id_column_name, how='left')
            assist_df_pb_subset['skill_id'] = assist_df_pb_subset['meta-kc']
            assist_df_pb_subset = assist_df_pb_subset.drop(columns=['meta-kc'])
            assist_df_pb_subset.to_csv(os.path.join(cluster_folder, csv_filename + '_dim_method_' + dim_method + '_dims_' + str(n_component) + '_n' + str(n_clusters) + '.csv'), index=False)

        # Optional: Save the t-SNE results as well
        tsne_output_path = os.path.join(cluster_folder, 'tsne_results.pt')
        torch.save(torch.tensor(skills_tsne), tsne_output_path)
        print(f"Saved t-SNE results to {tsne_output_path}")
