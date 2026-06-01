import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from gollum.featurization.text import get_tokens, get_huggingface_embeddings

# One-hot encoding now lives in gollum.featurization.protein (single source of
# truth, no matplotlib import cost). Re-exported here for backward compatibility.
from gollum.featurization.protein import one_hot_encode_sequences, one_hot_matrix


def viz_distances(data, split=False, onehot=False, enz_split=None):
    """
    This function visualizes the pairwise distances between various 
    protein sequences in the ESM embedding space. It generates a histogram 
    to show the distribution of these distances. It includes a flag to 
    indicate whether the distances are between the pairs in the whole 
    dataset or split separated: train-train, train-test and test-test pairs.
    """

    if onehot:
        sequences = data['sequence'].tolist()
        embeddings = one_hot_encode_sequences(sequences).reshape(len(sequences), -1)
    else:
        # Embed the sequences using ESM and compute pairwise distances
        sequences = data['sequence'].tolist()
        embeddings = get_huggingface_embeddings(sequences, model_name="facebook/esm2_t33_650M_UR50D", max_length=1280)
    if split:
        # Assuming data is a DataFrame with 'split' column indicating train/test
        
        train_embeddings = embeddings[data['set'] == 'train']
        test_embeddings = embeddings[data['set'] == 'test']

        # Compute distances for train-train, train-test, and test-test pairs
        train_distances = cdist(train_embeddings, train_embeddings, metric='euclidean')
        test_distances = cdist(test_embeddings, test_embeddings, metric='euclidean')
        cross_distances = cdist(train_embeddings, test_embeddings, metric='euclidean')

        # Plot histograms for each type of distance
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(train_distances.flatten(), bins=20, alpha=0.7, color='blue')
        plt.title('Train-Train Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.subplot(1, 3, 2)
        plt.hist(test_distances.flatten(), bins=20, alpha=0.7, color='orange')
        plt.title('Test-Test Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.hist(cross_distances.flatten(), bins=20, alpha=0.7, color='green')
        plt.title('Train-Test Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Save the plots
        plt.savefig(f"distance_distribution_split_onehot_{enz_split}.png" if onehot else f"distance_distribution_split_esm2_{enz_split}.png")
    else:
        distances = cdist(embeddings, embeddings, metric='euclidean')
        print("Pairwise distances computed. Visualizing...")

        # Create a histogram of the distances
        plt.hist(distances.flatten(), bins=20, alpha=0.7, color='blue')
        plt.title('Distribution of Distances')
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # Save the plot
        plt.savefig(f"distance_distribution_onehot_{enz_split}.png" if onehot else f"distance_distribution_esm2_{enz_split}.png")

if __name__ == "__main__":
    files = [
        "data/flip2/alpha-amylase/close_to_far.csv",
        "data/flip2/alpha-amylase/one_to_many.csv",
        "data/flip2/nucB/two_to_many.csv",
        "data/flip2/trpB/one_to_many.csv",
        "data/flip2/ired/two_to_many.csv",
    ]
    for data_file in files:
        enz_split = data_file.split("/")[2] + "_" + data_file.split("/")[-1].split(".")[0]
        print(f"Loading data {enz_split}...")
        data = pd.read_csv(data_file)
        print(f"Number of sequences: {len(data['sequence'])}")
        print(f"Average sequence length: {data['sequence'].apply(len).mean():.2f}")
        viz_distances(data, split=True, onehot=True, enz_split=enz_split)
        viz_distances(data, split=True, onehot=False, enz_split=enz_split)