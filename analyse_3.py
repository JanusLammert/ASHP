import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import OSS
import Affinity_matrix as am
import clustering as c

def calculate_accuracy_unknown_order(confusion_matrix):
    """
    Calculate accuracy from a confusion matrix when the order of rows and columns is unknown.

    Parameters:
    confusion_matrix (list of lists): Confusion matrix where rows represent true labels and columns represent predicted labels.

    Returns:
    accuracy (float): Accuracy of the classification algorithm.
    """
    # Apply the Hungarian algorithm to find the optimal mapping
    true_indices, pred_indices = linear_sum_assignment(confusion_matrix, maximize=True)
    
    # Compute accuracy using the optimal mapping
    total_samples = np.sum(confusion_matrix)
    
    total_correct = 0
    for i in range(5):
        total_correct+=confusion_matrix[true_indices[i], pred_indices[i]]

    if total_samples == 0:
        return 0.0
    
    accuracy = total_correct / total_samples
    return accuracy

def evaluate_spectral_clustering(affinity_map, true_labels):
    n_components_range = range(1, 20)  # Try different numbers of components
    scores = []

    for n_components in n_components_range:
        spectral_model = SpectralClustering(n_clusters=5, n_components=n_components, affinity='precomputed')
        predicted_labels = spectral_model.fit_predict(affinity_map)
        score = np.mean(predicted_labels == true_labels)
        scores.append(score)

    plt.plot(n_components_range, scores, marker='o')
    plt.title('Spectral Clustering Evaluation')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def evaluate_spectral_clustering_2(affinity_map, true_labels):
    n_components_range = range(1, 20)  # Try different numbers of components
    accuracies = []

    for n_components in n_components_range:
        spectral_model = SpectralClustering(n_clusters=5, n_components=n_components, affinity='precomputed')
        clusters = spectral_model.fit_predict(affinity_map)
        tabel = np.zeros((5, 5), dtype=np.int16)

        num_class_1 = 21
        num_class_2 = 20
        num_class_3 = 16
        num_class_4 = 30
        num_class_5 = 24

        for i, v in enumerate(clusters):
            if i < num_class_1:
                tabel[0,v] += 1
            elif num_class_1 <= i and i < num_class_1 + num_class_2:
                tabel[1,v] += 1
            elif num_class_1 + num_class_2 <= i and i < num_class_1 + num_class_2 + num_class_3:
                tabel[2,v] += 1
            elif num_class_1 + num_class_2 + num_class_3 <= i and i < num_class_1 + num_class_2 + num_class_3 + num_class_4:
                tabel[3,v] += 1
            else:
                tabel[4,v] +=1

        accuracy = calculate_accuracy_unknown_order(tabel)
        accuracies.append(accuracy)

    plt.plot(n_components_range, accuracies, marker='o', color=(0,61/255,100/255))
    plt.title('Spectral Clustering Evaluation')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(min(n_components_range), max(n_components_range)+1, 2.0))
    plt.grid(True)
    plt.savefig('n_comp_analysis_2.svg')
    plt.show()

def generate_array():
    array_part_0 = np.zeros(21, dtype=int)
    array_part_1 = np.ones(20, dtype=int)
    array_part_2 = np.full(16, 2, dtype=int)
    array_part_3 = np.full(30, 3, dtype=int)
    array_part_4 = np.full(24, 4, dtype=int)
    
    result_array = np.concatenate((array_part_0, array_part_1, array_part_2, array_part_3, array_part_4))
    return result_array

Benedicts = generate_array()
infile='/home/janus/Uni/Masterarbeit/ASHP/data/run_MSE_w10_n25_t_.npy'

input_array = OSS.open_np(infile)
res = am.aff_matrix(input_array, mode='MSE', num_points=10)

evaluate_spectral_clustering_2(res,Benedicts)