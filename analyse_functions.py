import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import OSS
import Preprocessing as p

def calculate_affinity_statistics(class_assignment, affinity_matrix):
    class_assignment = np.asarray(class_assignment)
    num_classes = len(np.unique(class_assignment))
    class_affinity_sum = np.zeros(num_classes)
    class_point_count = np.zeros(num_classes)
    class_outer_affinity_sum = np.zeros(num_classes)
    class_outer_point_count = np.zeros(num_classes)

    for i in range(len(class_assignment)):
        class_index = class_assignment[i]
        class_affinity_sum[class_index] += np.sum(affinity_matrix[i][class_assignment == class_index])
        class_point_count[class_index] += np.sum(class_assignment == class_index)
        class_outer_affinity_sum[class_index] += np.sum(affinity_matrix[i][class_assignment != class_index])
        class_outer_point_count[class_index] += np.sum(class_assignment != class_index)

    class_avg_affinity = class_affinity_sum / class_point_count
    class_avg_outer_affinity = class_outer_affinity_sum / class_outer_point_count

    return class_avg_affinity, class_avg_outer_affinity

def visualize_affinity_matrix_2(affinity_matrix, clusters):
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
        if clusters[i]==0:
            colors.append((0,61/255,100/255))
        elif clusters[i]==1:
            colors.append((255/255, 201/255, 185/255))
        elif clusters[i]==2:
            colors.append((195/255,214/255,155/255))
        elif clusters[i]==3:
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


def visualize_affinity_matrix3(affinity_matrix, clusters, k=5):
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
        if clusters[i]==0:
            colors.append((0,61/255,100/255))
        elif clusters[i]==1:
            colors.append((255/255, 201/255, 185/255))
        elif clusters[i]==2:
            colors.append((195/255,214/255,155/255))
        elif clusters[i]==3:
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

def sort_affinity_map(affinity_map, labels):
    """
    Sorts the affinity map into different clusters based on the given labels while preserving symmetry.

    Parameters:
        affinity_map (numpy.ndarray): Affinity map with shape (N, N), where N is the number of data points.
        labels (numpy.ndarray): Array of cluster labels for each data point.

    Returns:
        numpy.ndarray: Sorted affinity map where rows and columns are grouped by cluster labels.
    """
    unique_labels = np.unique(labels)
    sorted_affinity_map = np.zeros_like(affinity_map)

    # Create a permutation to sort rows and columns simultaneously
    permutation = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        permutation.extend(indices)
    permutation = np.array(permutation)

    # Apply permutation to both rows and columns
    sorted_affinity_map = affinity_map[permutation][:, permutation]

    return sorted_affinity_map

def count_frequencies(arr):
    # Check if the array is empty
    if arr.size == 0:
        return [0]
    
    # Find the maximum value in the array to determine the size of the frequency array
    max_value = int(np.max(arr))
    
    # Create a list of zeros with length max_value + 1
    frequency = [0] * (max_value + 1)
    
    # Count the frequency of each value in the input array
    for value in arr:
        frequency[int(value)] += 1
    
    return frequency

def generate_pattern_list(lst):
    # Determine the length of the input list
    length = len(lst)
    
    # Create the output list with 'P1', 'P2', ..., 'P(n-1)', ''
    output_list = [f'P{i+1}' for i in range(length - 1)] + ['']
    
    return output_list

def plot_matrix(matrix, labels, cmap='viridis', path='/p/scratch/cvsk18/lammert1/240126_fibrils_ASHP/images/', name=''):
    
    number_of_elements = count_frequencies(labels)
    number_of_elements.insert(0, 0)
    labeling = generate_pattern_list(number_of_elements)

    # Plot the matrix
    plt.imshow(matrix, vmax=np.max(matrix[matrix != 1]), cmap=cmap)
    plt.colorbar(label='Value')
    
    # Customize the x-axis labels
    plt.xticks(number_of_elements, labeling)
    
    # Customize the y-axis labels
    plt.yticks(number_of_elements, labeling, rotation=90)
    
    plt.savefig(path + name + '.svg')

    plt.close()

def plot_image_histogram(image, path='/p/scratch/cvsk18/lammert1/240126_fibrils_ASHP/images/', name=''):
    """
    Plots the histogram of an image with pixel values between 0 and 1.
    
    Parameters:
        image (numpy.ndarray): The input image with pixel values between 0 and 1.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input image must be a numpy array.")
        
    if np.min(image) < 0 or np.max(image) > 1:
        raise ValueError("Pixel values must be between 0 and 1.")

    # Flatten the image array to get the pixel values
    pixel_values = image.flatten()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(pixel_values, bins=50, range=(0, 1), color=(0/255, 61/255, 100/255), alpha=0.7)
    plt.xlabel('Affinity')
    plt.ylabel('Count')
    plt.savefig(path + name + '.svg')
    plt.close()  # Use plt.close() to avoid attempting to show the plot


def plot_image_histogram_diff(image, path='/p/scratch/cvsk18/lammert1/240126_fibrils_ASHP/images/', name=''):
    """
    Plots the histogram of an image with pixel values between 0 and 1.
    
    Parameters:
        image (numpy.ndarray): The input image with pixel values between 0 and 1.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The input image must be a numpy array.")
        
    if np.min(image) < -1 or np.max(image) > 1:
        raise ValueError("Pixel values must be between 0 and 1.")

    # Flatten the image array to get the pixel values
    pixel_values = image.flatten()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(pixel_values, bins=50, range=(-1, 1), color=(0/255, 61/255, 100/255), alpha=0.7)
    plt.xlabel('Difference')
    plt.ylabel('Count')
    plt.savefig(path + name + '.svg')
    plt.show()

def separate_and_save_images(file, labels, output_prefix):
    """
    Opens a stack of images from an MRC file, separates them based on labels,
    and saves the separated images.

    Parameters:
        file (str): The path to the input MRC file.
        labels (list of int): The list of labels corresponding to the clustering of images.
        output_prefix (str): The prefix for the output MRC files.
    """
    # Open the MRC file
    image_stack = OSS.open_mrc(file)
    
    good, bad = p.first_selection(image_stack)
    del bad
    image_stack = good

    if image_stack is None:
        raise ValueError("Failed to open MRC file.")
    
    # Check the dimensions of the image stack
    if len(image_stack.shape) != 3:
        raise ValueError("Input MRC file must contain a stack of images.")
    
    # Ensure the number of labels matches the number of images in the stack
    num_images = image_stack.shape[0]
    if len(labels) != num_images:
        print(f"Number of labels: {len(labels)}; Number of images: {num_images}")
        raise ValueError("The number of labels must match the number of images in the stack.")
    
    # Create a dictionary to store separated images
    separated_images = {}
    
    for i, label in enumerate(labels):
        if label not in separated_images:
            separated_images[label] = []
        separated_images[label].append(image_stack[i])
    
    # Save the separated images
    for label, images in separated_images.items():
        output_file = f"{output_prefix}_label_{label}"
        # Stack images along the first dimension to form a 3D array
        image_stack_for_label = np.stack(images)

        # Convert the data type to float32 (or another supported type)
        image_stack_for_label = image_stack_for_label.astype(np.float32)

        OSS.save_mrc(output_file, image_stack_for_label)
        print(f"Saved {len(images)} images to {output_file}")
