import numpy as np
import matplotlib.pyplot as plt
import force_atlas as fa

def plot_affinity_matrix(size=10, max_affinity=0.28):
    # Generate random affinity matrix with values between 0 and max_affinity
    affinity_matrix = (fa.sym(np.random.rand(size, size) * max_affinity))**1.2

    for i in range(size):
        affinity_matrix[i,i]=1
    
    # Plot the affinity matrix
    plt.imshow(affinity_matrix, cmap='viridis', vmin=0, vmax=max_affinity)
    
    # Add labels to the axes
    plt.xticks(ticks=np.arange(size), labels=[f'Class {i+1}' for i in range(size)], rotation=45)
    plt.yticks(ticks=np.arange(size), labels=[f'Class {i+1}' for i in range(size)])
    
    # Add colorbar
    plt.colorbar(label='Affinity')
    
    # Add title
    plt.title('Affinity Matrix')
    
    # Show plot
    plt.show()



# Example usage:
plot_affinity_matrix()
