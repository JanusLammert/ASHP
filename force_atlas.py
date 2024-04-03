import numpy as np
#import networkx as nx
#from forceatlas2 import ForceAtlas2
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

def visualize_affinity_matrix(affinity_matrix):
    """
    Visualize affinity matrix using Multidimensional Scaling (MDS).
    
    Parameters:
    - affinity_matrix: numpy array, affinity matrix where affinity_matrix[i, j] represents
                       the affinity between point i and point j.
                       
    Returns:
    None (plots the visualization).
    """
    # Ensure the affinity matrix is symmetric
    assert (affinity_matrix == affinity_matrix.T).all(), "Affinity matrix must be symmetric"
    
    # Calculate dissimilarity matrix from affinity matrix
    dissimilarity_matrix = 1 - affinity_matrix / np.max(affinity_matrix)
    
    # Perform MDS to reduce dimensionality to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed')
    coordinates = mds.fit_transform(dissimilarity_matrix)
    
    # Plot the points
    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='b', s=50)
    
    # Annotate points with their indices
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i+1), fontsize=12)


    plt.title('Multidimensional Scaling (MDS) Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

def sym(Arr):
    x,y = np.shape(Arr)
    if x!=y:
        print("ERROR: Array not symmatrizable")
        return Arr
    else:
        for i in range(x):
            for j in range(i+1,y):
                Arr[i,j]=(Arr[i,j]+Arr[j,i])/2
                Arr[j,i]=Arr[i,j]
        return Arr


def visualize_affinity_matrix_2(affinity_matrix):
    """
    Visualize affinity matrix using Multidimensional Scaling (MDS).

    Parameters:
    - affinity_matrix: numpy array, affinity matrix where affinity_matrix[i, j] represents
                       the affinity between point i and point j.

    Returns:
    None (plots the visualization).
    """
    # Ensure the affinity matrix is symmetric
    assert (affinity_matrix == affinity_matrix.T).all(), "Affinity matrix must be symmetric"

    # Calculate dissimilarity matrix from affinity matrix
    dissimilarity_matrix = 1 - affinity_matrix / np.max(affinity_matrix)

    # Perform MDS to reduce dimensionality to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed')
    coordinates = mds.fit_transform(dissimilarity_matrix)

    # Assign colors based on index ranges
    num_points = len(affinity_matrix)
    colors = []
    for i in range(num_points):
        if i < 21:
            colors.append((0,61/255,100/255))
        elif i < 21 + 20:
            colors.append((255/255, 201/255, 185/255))
        elif i < 21 + 20 + 16:
            colors.append((195/255,214/255,155/255))
        elif i < 21 + 20 + 16 + 30:
            colors.append((62/255, 137/255, 137/255))
        else:
            colors.append((117/255, 109/255, 84/255))

    # Plot the points with colors based on index ranges
    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=50)

    plt.title('Multidimensional Scaling (MDS) Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('MDS.svg')
    plt.show()


def visualize_affinity_matrix3(affinity_matrix, k=5):
    """
    Visualize affinity matrix using Multidimensional Scaling (MDS) considering only the k nearest neighbors,
    with colors based on the indices of the points.

    Parameters:
    - affinity_matrix: numpy array, affinity matrix where affinity_matrix[i, j] represents
                       the affinity between point i and point j.
    - k: int, optional, number of nearest neighbors to consider.

    Returns:
    None (plots the visualization).
    """
    # Ensure the affinity matrix is symmetric
    assert (affinity_matrix == affinity_matrix.T).all(), "Affinity matrix must be symmetric"

    # Calculate dissimilarity matrix from affinity matrix
    dissimilarity_matrix = 1 - affinity_matrix / np.max(affinity_matrix)

    # Find k nearest neighbors for each data point
    nn = NearestNeighbors(n_neighbors=k+1)  # Include the point itself
    nn.fit(dissimilarity_matrix)
    indices = nn.kneighbors(return_distance=False)[:, 1:]  # Exclude the point itself

    # Compute distances considering only k nearest neighbors
    reduced_dissimilarity_matrix = np.zeros_like(dissimilarity_matrix)
    for i, neighbors in enumerate(indices):
        reduced_dissimilarity_matrix[i, neighbors] = dissimilarity_matrix[i, neighbors]
        reduced_dissimilarity_matrix[neighbors, i] = dissimilarity_matrix[neighbors, i]

    # Perform MDS to reduce dimensionality to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed')
    coordinates = mds.fit_transform(reduced_dissimilarity_matrix)

    # Assign colors based on index ranges
    num_points = len(affinity_matrix)
    colors = []
    for i in range(num_points):
        if i < 21:
            colors.append((0,61/255,100/255))
        elif i < 21 + 20:
            colors.append((255/255, 201/255, 185/255))
        elif i < 21 + 20 + 16:
            colors.append((195/255,214/255,155/255))
        elif i < 21 + 20 + 16 + 30:
            colors.append((62/255, 137/255, 137/255))
        else:
            colors.append((117/255, 109/255, 84/255))

    # Plot the points with colors based on index ranges
    plt.figure(figsize=(8, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=50)

    plt.title('Multidimensional Scaling (MDS) Visualization considering only {} nearest neighbors'.format(k))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig('MDS.svg')
    plt.show()
