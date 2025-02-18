import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

np.random.seed(3)

def generate_affinity_matrix(n_points):
    """
    Generate a synthetic affinity matrix.

    Parameters:
        n_points (int): Number of data points.

    Returns:
        numpy.ndarray: Affinity matrix of shape (n_points, n_points).
    """
    # Generate random data points
    data_points = np.random.rand(n_points, 1)
    
    # Compute pairwise distances
    distances = np.abs(data_points - data_points.T)
    
    # Compute affinity matrix (Gaussian kernel)
    affinity_matrix = np.exp(-distances ** 2)
    
    # Ensure the diagonal is all one (optional)
    np.fill_diagonal(affinity_matrix, 1)
    
    return affinity_matrix **2

def Kruskal_clustering(data):
    if len(np.shape(data)) != 2:
        print("Input for Kruskal clustering should be a two-dimensional array.")
        return
    num_data_points, temp = np.shape(data)
    if num_data_points != temp:
        print("The input for Kruskal clustering should be of the shape n x n.")
        return
    del temp
    
    transformed_data = transform_array(data)

    num_sim = np.shape(transformed_data)[1]

    sorted_transformed_data = sort_array_by_value(transformed_data)

    clusters = []
    cluster = np.arange(num_data_points)
    cluster_copy = np.copy(cluster)
    clusters.append(cluster_copy)

    linkage = []
    
    ID = num_data_points

    for i in range(num_sim):
        if cluster[int(sorted_transformed_data[1,i])] != cluster[int(sorted_transformed_data[2,i])]:
            temp = cluster[int(sorted_transformed_data[1,i])]
            target = cluster[int(sorted_transformed_data[2,i])]
            linkage.append([temp, target, 1 - sorted_transformed_data[0,i], target])
            for index, value in enumerate(cluster):
                if value == temp or value == target:
                    cluster[index] = ID
            ID+=1
            cluster_copy = np.copy(cluster)
            clusters.append(cluster_copy)

    return clusters, linkage

def transform_array(input_array):
    """
    Transform an n x n array into a 3 x (n^2 - n) array.

    Parameters:
        input_array (numpy.ndarray): Input array of shape (n, n).

    Returns:
        numpy.ndarray: Transformed array of shape (3, n^2 - n).
    """
    n = input_array.shape[0]
    output_array = np.zeros((3, n**2 - n))
    
    index = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                output_array[1, index] = i  # First index
                output_array[2, index] = j  # Second index
                output_array[0, index] = input_array[i, j]  # Corresponding value
                index += 1
    
    return output_array

def sort_array_by_value(output_array):
    """
    Sort the 3 x (n^2 - n) array by the first value.

    Parameters:
        output_array (numpy.ndarray): Transformed array of shape (3, n^2 - n).

    Returns:
        numpy.ndarray: Sorted array by the first value.
    """
    sorted_indices = np.argsort(output_array[0])[::-1]
    sorted_array = output_array[:, sorted_indices]
    return sorted_array

def plot_dendrogram(linkage_matrix, labels=None):
    # Plot the dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10)
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.title('Dendrogram')
    plt.show()

def rename_clusters(clusters):
    dic=[]
    for i in range(len(clusters)):
        if clusters[i] not in dic:
            dic.append(clusters[i])
    dic = np.asarray(dic)
    for i in range(len(clusters)):
        clusters[i]=np.where(dic==clusters[i])
    return clusters

if __name__ == "__main__":
    # Example usage:
    n_points = 10
    affinity_matrix = generate_affinity_matrix(n_points)
    print(affinity_matrix)

    out, jumps = Kruskal_clustering(affinity_matrix)
    print(out)
    print(jumps)

    plot_dendrogram(jumps)
