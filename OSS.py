#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:43:25 2022

@author: Janus Lammert ☺

Functions
---------
open_mrc : Opens an .mrc file to a numpy Array

show_img : Shows a Stack of images

save_np : save a numpy Array to a .npy file

open_np : opens a .npy file to an numpy Array

Display_conv_data : Displays the Data of the convolution or the MSE

"""

import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import math
import os
import time

if __name__ == "__main__":
    print("execute main.py")

def open_mrc(file):
    """
    Opens mrcs files into an array

    Parameters
    ----------
    file : string
        Path to the mrcs file.

    Returns
    -------
    data : Array of float32
        3D-Array with the images. The first index describes index of the Image and the following indices descirbe the postition of the pixels.

    """
    print('Open files')
    with mrcfile.open(file) as mrc:
        data = mrc.data
        return data
    
def save_mrc(name, data):
    """
    Saves np-Arrays into .mrc-files.

    Parameters
    ----------
    name : string
        Path to the mrcs file.
    data : 3D-Array
        Data for the .mrc-files
    """
    print('Open files')
    with mrcfile.new(name + ".mrcs",overwrite=True) as mrc:
        mrc.set_data(data)

def show_img(image, save=False, path='Images/Img_temp.png', dpi=800, normalized=True, cmap='gray'):
    """
    Shows a variable amount of images and saves them if requiered. (By default in a seperate folder)

    Parameters
    ---------- bad = p.first_selection(OSS.open_mrc(input_img))
del bad
OSS.sho
    image : Array (2D or 3D)
        Images stacked with the first index indexing the images.
    save : Boolean, optional
        If the Image should be saved set it to True otherwise False. The default is False.
    path : String, optional
        Image name. If wanted one can add a directory or a path in front. If the folder doesn't exist it gets created. The default is 'Images/Img_temp.png'.
    dpi : Integer, optional
        Resolution in Dots per Inch. The default is 800.
    normalized : Boolean, optional
        Should the scale be the same for every image?
    cmap : String, optional
        Colourmap

    Returns
    -------
    None.

    """
    dim = len(np.shape(image))
    if dim == 2:
        plt.figure()
        if normalized:
            plt.imshow(image, cmap=cmap, vmin=np.min(
                image), vmax=np.max(image))
        else:
            plt.imshow(image, cmap=cmap)
        if save:
            Path(path).mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=dpi)
    elif dim == 3:
        lenz, leny, lenx = np.shape(image)
        xdim = 5
        if lenz <= 20:
            ydim = math.ceil(lenz/xdim)
            fig = plt.figure()
            for i in range(1, lenz+1):
                fig.add_subplot(ydim, xdim, i)
                if normalized:
                    plt.imshow(image[i-1, :, :], cmap=cmap,
                               vmin=np.min(image), vmax=np.max(image))
                else:
                    plt.imshow(image[i-1, :, :], cmap=cmap)
                plt.axis('off')
                plt.title(i-1)
            if save:
                Path(path).mkdir(parents=True, exist_ok=True)
                plt.savefig(path, dpi=dpi)
            plt.show()
        else:
            num_figs = math.ceil(lenz/20)
            counter2 = 0
            for j in range(num_figs-1):
                fig = plt.figure()
                counter = 1
                ydim = 4
                for i in range(j*20+1, (j+1)*20+1):
                    fig.add_subplot(ydim, xdim, counter)
                    if normalized:
                        plt.imshow(image[i-1, :, :], cmap=cmap,
                                   vmin=np.min(image), vmax=np.max(image))
                    else:
                        plt.imshow(image[i-1, :, :], cmap=cmap)
                    plt.axis('off')
                    plt.ylabel(str(i-1))
                    counter += 1
                if save:
                    counter2 += 1
                    save_Path = Path(path).parent
                    save_Path.mkdir(parents=True, exist_ok=True)
                    plt.savefig(os.path.splitext(path)[
                                0] + "(" + str(counter2) + ").png", dpi=dpi)
            ydim = math.ceil((lenz % 20)/xdim)
            fig = plt.figure()
            for i in range(1, (lenz % 20)+1):
                fig.add_subplot(ydim, xdim, i)
                if normalized:
                    plt.imshow(image[num_figs*20+i-21, :, :], cmap=cmap,
                               vmin=np.min(image), vmax=np.max(image))
                else:
                    plt.imshow(image[num_figs*20+i-21, :, :], cmap=cmap)
                plt.axis('off')
                plt.ylabel(num_figs*20+i-21)
            if save:
                save_Path = Path(path).parent
                save_Path.mkdir(parents=True, exist_ok=True)
                plt.savefig(os.path.splitext(path)[
                            0] + "(" + str(num_figs) + ").png", dpi=dpi)
            plt.show()
    else:
        print('ERROR: Input a 2 or 3 dimensional Array into the show_img() function')
        
def save_np(Arr, name='Results', spec='date', path='Results/'):
    """
    Saves numpyArrays as .npy-files

    Parameters
    ----------
    Arr : n-dimensional numpy array
        The array that is supposed to be saved.
    name : String, optional
        Name of the file. The default is 'Results'.
    spec : String, optional
        Specify the Result. (Is also in the title) The default is 'date'.
    path : String, optional
        Path where the file should be saved. The default is 'Results/'.

    Returns
    -------
    None.

    """
    if spec == 'date':
        timestr = time.strftime("%Y%m%d-%H%M%S")
        pathname = path + name + '_' + timestr
        save_Path = Path(pathname).parent
        save_Path.mkdir(parents=True, exist_ok=True)
        np.save(pathname, Arr)
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        pathname = path + name + '_' + spec
        save_Path = Path(pathname).parent
        save_Path.mkdir(parents=True, exist_ok=True)
        np.save(pathname, Arr)
    
def open_np(File):
    """
    opens an .npy-file to an numpy Array.

    Parameters
    ----------
    File : String
        Path and Filename to import.

    Returns
    -------
    Arr : n-dimensional np Array
        The Array that gets imported.

    """
    Arr = np.load(File)
    return Arr
    
def Display_conv_data(data, ind1, ind2, save=False,show=False, name='conv_data', path='Uni/Masterarbeit/fibrils/Images/'):
    """
    Displays Data from MSE or convolution calculations

    Parameters
    ----------
    data : 2D-Array
        The results of the MSE or convolution calculations that one wants to 
        display sliced with the first two indiced beeing given by ind1 and ind2.
    ind1 : Int
        see above.
    ind2 : Int
        see above.

    Returns
    -------
    None.

    """
    shape = np.shape(data)
    scale = np.linspace(1, shape[0], shape[0])
    ycoord= (data[:,1]+95)%200
    plotdataX = [scale, scale, data[:, 0], data[:, 2]]
    plotdataY = [data[:, 0], data[:, 2], ycoord, ycoord]
    axisX = ['Index of Window', 'Index of Window', 'max value', 'x-position']
    axisY = ['max value', 'x-position', 'y-position', 'y-position']
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(f'Window from {ind1} and applied to {ind2}')
    fig.tight_layout(pad=2.0)
    i = 0
    for row in ax:
        for col in row:
            col.scatter(plotdataX[i], plotdataY[i], color=(0,61/255,100/255))
            col.set(xlabel=axisX[i], ylabel=axisY[i])
            i += 1

    # Setting x and y limits for the fourth graph
    ax[1, 1].set_xlim(0, 200)
    ax[1, 1].set_ylim(0, 200)

    if save:
        plt.savefig(path + name, dpi=800)
    if show:
        plt.show()
    
def display_classes(images, labels, save=False, path='Images/', name='temp', frame_width=6, color='default'):
    frame_colors=['red', 'blue', 'green', 'yellow', 'purple', 
                                        'orange', 'pink', 'brown', 'gray', 'cyan']
    frame_colors_cd=[(0,61,100),(255, 201, 185), (195,214,155), (62, 137, 137), (117, 109, 84)]
    chunk_size = 20
    chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
    for chunk_index, chunk in enumerate(chunks):
        figure, axes = plt.subplots(5, 4, figsize=(20, 20))
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i < len(chunk):
                ax.imshow(chunk[i], cmap='gray')
                ax.set_axis_off()
                ax.set_title("Class %d" % (labels[chunk_index * chunk_size + i + 1]), fontsize=20)
                if color=='default':
                    rect = patches.Rectangle((0, 0), chunk[i].shape[0], chunk[i].shape[1],
                                         linewidth=frame_width, edgecolor=frame_colors[labels[chunk_index * chunk_size + i + 1]], facecolor='none')
                elif color=='juelich':
                    rect = patches.Rectangle((0, 0), chunk[i].shape[0], chunk[i].shape[1],
                                         linewidth=frame_width, edgecolor=frame_colors_cd[labels[chunk_index * chunk_size + i + 1]], facecolor='none')
                else:
                    print("Unknown paramter for color")
                ax.add_patch(rect)
        plt.tight_layout()
        if save:
            pathname = path + name + '.png'
            save_Path = Path(pathname).parent
            save_Path.mkdir(parents=True, exist_ok=True)
            plt.savefig((path + name + '('+ str(chunk_index) + ').png'), dpi=400)
        plt.show()
        
        
def display_classes2(images, labels, save=False, path='Images/', name='temp', frame_width=10, color='default'):
    frame_colors=['red', 'blue', 'green', 'yellow', 'purple', 
                                        'orange', 'pink', 'brown', 'gray', 'cyan']
    frame_colors_cd=[(0,61/255,100/255),(255/255, 201/255, 185/255), (195/255,214/255,155/255), (62/255, 137/255, 137/255), (117/255, 109/255, 84/255)]
    chunk_size = 20
    num_img = math.ceil(len(images)/chunk_size)
    missing_labels = np.zeros(num_img*chunk_size, dtype=int)
    missing_labels[:len(labels)] = labels
    labels = missing_labels
    chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
    for chunk_index, chunk in enumerate(chunks):
        figure, axes = plt.subplots(5, 4, figsize=(20, 20))
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            if i < len(chunk):
                ax.imshow(chunk[i], cmap='gray')
                ax.set_axis_off()
                ax.set_title("Class %d" % (labels[chunk_index * chunk_size + i + 1]), fontsize=20)
                if color=='default':
                    rect = patches.Rectangle((0, 0), chunk[i].shape[0], chunk[i].shape[1],
                                         linewidth=frame_width, edgecolor=frame_colors[labels[chunk_index * chunk_size + i + 1]], facecolor='none')
                elif color=='juelich':
                    rect = patches.Rectangle((0, 0), chunk[i].shape[0], chunk[i].shape[1],
                                         linewidth=frame_width, edgecolor=frame_colors_cd[labels[chunk_index * chunk_size + i + 1]], facecolor='none')
                else:
                    print("Unknown paramter for color")
                ax.add_patch(rect)
            else:
                ax.set_axis_off()
                ax.set_visible(False)
        plt.tight_layout()
        if save:
            pathname = path + name + '.png'
            save_Path = Path(pathname).parent
            save_Path.mkdir(parents=True, exist_ok=True)
            plt.savefig((path + name + '('+ str(chunk_index) + ').png'), dpi=400)
        plt.show()
        
