import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def read_data(file_path):
    data = {'MSD': {'laplacian': [], 'plane': []}, 'convolution': {'laplacian': [], 'plane': []}}
    with open(file_path, 'r') as f:
        for line in f:
            columns = line.strip().split('\t')
            if len(columns) == 6:
                first_col, second_col, col3, col4, col5, col6 = columns
                if first_col in ['MSD', 'convolution'] and second_col in ['laplacian', 'plane'] and col5 in ['3', '5', '10']:
                    data[first_col][second_col].append((int(col3), int(col5), float(col6)))
    return data

def plot_figure(ax, data, title):
    for value, size, color in data:
        ax.scatter(value, size, color=color, s=100)
    ax.set_title(title)

def plot_data(data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    axs = axs.flatten()

    all_colors = [color for subdata in data.values() for values in subdata.values() for _, _, color in values]
    min_color = min(all_colors)
    max_color = max(all_colors)

    cmap = plt.get_cmap('viridis')
    normalize = Normalize(vmin=min_color, vmax=max_color)

    for i, (key1, subdata) in enumerate(data.items()):
        for j, (key2, values) in enumerate(subdata.items()):
            ax = axs[i*2 + j]
            colors = [cmap(normalize(color)) for _, _, color in values]
            plot_figure(ax, values, f'{key1.capitalize()} and {key2.capitalize()}')
            ax.set_xlabel('Column 3')
            ax.set_ylabel('Value')
            sm = ScalarMappable(cmap=cmap, norm=normalize)
            sm.set_array([])
            plt.colorbar(sm, ax=ax)

    plt.show()

if __name__ == "__main__":
    file_path = input("Enter the path to the .dat file: ")
    data = read_data(file_path)
    print(data)
    plot_data(data)

