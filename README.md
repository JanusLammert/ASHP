# ASHP: Automated Separation of Helical Polymers

**ASHP** is a tool for the **Automated Separation of 2D Classes** based on the 2D-class-averages, combined with **CHEP** [1].

## Prerequisites

Before you start, ensure you have the following installed:

1. **Conda** - Used for environment management.

## Installation

Follow these steps to get ASHP up and running:

1. Install the conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. Download all the Python scripts for ASHP.

3. Make the scripts executable:
   ```bash
   chmod +x ASHP
   ```

## Guide

1. Activate the `ASHP` conda environment:
   ```bash
   conda activate ASHP
   ```

2. Run the help command to learn about available options:
   ```bash
   ./ASHP --help
   ```

3. Deactivate the conda environment when done:
   ```bash
   conda deactivate
   ```

## How to Use

### REQUIRED ARGUMENTS

- `-i, --infile`  
  Path to the 2D classification Relion `.mrc` file.

- `-is, --infile_star`  
  Path to the particle or data star file.

- `-o, --output`  
  Path for the output star file.

---

### OPTIONAL ARGUMENTS

- `-c, --cluster_num`  
  Number of clusters for the clustering algorithm.  
  *Can be a list. Example: `-c 2 3 4 5`.*

- `-og, --outpath_graphs`  
  Directory name for output images.

- `-m, --metric`  
  Metric for distance calculation. Options: `MSE`, `conv`.  
  *(Default: MSE)*

- `-w, --window_size`  
  Window size, ideally ~10% of the box size.  
  *(Default: 26)*

- `-n, --num_window`  
  Number of windows used.  
  *(Default: 5)*

- `-t, --topnum`  
  Number of top-ranked windows considered for classification.  
  *(Default: 5)*

- `-p, --parallelise`  
  Number of CPU cores to use. If not set, it uses all available cores.  
  *(Default: -1)*

- `-s, --size`  
  Size reduction in Y direction to reduce computation time.  
  *(Default: 60)*

- `-r, --reduction`  
  Reduction of window size relative to the class image.  
  *(Default: 5)*

- `-f, --factor`  
  Weight between CHEP and ASHP.  
  Lower values weight ASHP more strongly; higher values weight CHEP more strongly.  
  *(Default: 1)*

- `-V, --version`  
  Print ASHP version and exit.

---

### EXPERT OPTIONS (KEEP AT DEFAULT)

- `-a, --algorithm`  
  Algorithm used for clustering. Options: `Spectral_clustering`, `kmeans`.  
  *(Default: Spectral_clustering)*

- `-l, --laplacian`  
  Should the Laplacian be calculated?  
  *(Default: False)*

- `-fu, --func`  
  Toggle window function.  
  *(Default: False)*

---

## References

[1] **Pothula, K. R., et al. (2019).**  
   "Clustering cryo-EM images of helical protein polymers for helical reconstructions." *Ultramicroscopy*, 203: 132-138.  
   Helical protein polymers are often dynamic and complex assemblies, with many conformations and flexible domains possible within the helical assembly. During cryo-electron microscopy reconstruction, classification of the image data into homogeneous subsets is a critical step for achieving high resolution, resolving different conformations, and elucidating functional mechanisms. Hence, methods aimed at improving the homogeneity of these datasets are becoming increasingly important. In this paper, we introduce a new algorithm that uses results from 2D image classification to sort 2D classes into groups of similar helical polymers. We show that our approach is able to distinguish helical polymers that differ in conformation, composition, and helical symmetry. Our results on test and experimental cases — actin filaments and amyloid fibrils — illustrate how our approach can be useful to improve the homogeneity of a data set. This method is exclusively applicable to helical polymers and other limitations are discussed.

## License

**Automated Separation of Helical Polymers (ASHP)**  
Copyright (C) 2025 Janus Lammert

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License** as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but **WITHOUT ANY WARRANTY**; without even the implied warranty of **MERCHANTABILITY** or **FITNESS FOR A PARTICULAR PURPOSE**. See the **GNU General Public License** for more details.

You should have received a copy of the **GNU General Public License** along with this program. If not, see <https://www.gnu.org/licenses/>.
