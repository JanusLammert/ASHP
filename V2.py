#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:48:02 2022

@author: Janus Lammert ☺
"""

import mrcfile
import numpy as np
import scipy as sp
import math
import tqdm
import os
import sys
from argparse import ArgumentParser


VERSION = '0.9'
progname = os.path.basename(sys.argv[0])
parser = ArgumentParser(description='Beschreibung was der Code macht')

class TwoDclasses(object):
    
    def __init__(self, fname):
        self.fname = fname
        self.class_imges = None
        self.class_windows = None
        
    def load_img(self):
        print('Open files')
        with mrcfile.open(self.fname) as mrc:
            data = mrc.data
            self.class_imges = data
            return data
            
    def create_windows(self, windowsize=15, num_win=50, dis_win=0, WindowFunction='flattop'):
        if dis_win == 0:
            lenz, leny, lenx = np.shape(self.class_imges)
            dis_win = (lenx - windowsize)/num_win
        elif num_win != 32 and dis_win != 0:
            print('ERROR: Pleas only set num_win or dis_win')
        else:
            num_win = math.floor((lenx - windowsize)/dis_win)
        if WindowFunction == 'flattop':
            window_func = sp.signal.windows.flattop(windowsize)
        elif WindowFunction == 'flat':
            window_func = np.ones(windowsize)
        else:
            print('Please use a viable windowfunction')
        result = np.zeros((lenz, num_win, leny, windowsize))
        for wn in tqdm.tqdm(range(num_win), desc = 'Calculate windowcutouts'):
            for i in range(lenz):
                for y in range(leny):
                    result[i, wn, y, :] = self.class_imges[i, y, int(
                        wn*dis_win):int(wn*dis_win+windowsize)]*window_func
        self.class_windows = result()
        return result
    
    def reduce_size(self, data, red_x=0, red_y=0):
        dim = len(np.shape(data))
        if red_x != 0 and red_y != 0:
            if dim == 2:
                reduced_img = data[red_y:-1*red_y, red_x:-1*red_x]
            elif dim == 3:
                reduced_img = data[:, red_y:-1*red_y, red_x:-1*red_x]
            elif dim == 4:
                reduced_img = data[:, :, red_y:-1*red_y, red_x:-1*red_x]
            else:
                print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
        elif red_x == 0 and red_y != 0:
            if dim == 2:
                reduced_img = data[red_y:-1*red_y, :]
            elif dim == 3:
                reduced_img = data[:, red_y:-1*red_y, :]
            elif dim == 4:
                reduced_img = data[:, :, red_y:-1*red_y, :]
            else:
                print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
        elif red_x != 0 and red_y == 0:
            if dim == 2:
                reduced_img = data[:, red_x:-1*red_x]
            elif dim == 3:
                reduced_img = data[:, :, red_x:-1*red_x]
            elif dim == 4:
                reduced_img = data[:, :, :, red_x:-1*red_x]
            else:
                print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
        return reduced_img
    
    def reduce_window_size(self, red_x=0, red_y=75):
        dim = len(np.shape(self.class_windows))
        if red_x != 0 and red_y != 0:
            if dim == 2:
                reduced_img = self.class_windows[red_y:-1*red_y, red_x:-1*red_x]
            elif dim == 3:
                reduced_img = self.class_windows[:, red_y:-1*red_y, red_x:-1*red_x]
            elif dim == 4:
                reduced_img = self.class_windows[:, :, red_y:-1*red_y, red_x:-1*red_x]
            else:
                print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
        elif red_x == 0 and red_y != 0:
            if dim == 2:
                reduced_img = self.class_windows[red_y:-1*red_y, :]
            elif dim == 3:
                reduced_img = self.class_windows[:, red_y:-1*red_y, :]
            elif dim == 4:
                reduced_img = self.class_windows[:, :, red_y:-1*red_y, :]
            else:
                print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
        elif red_x != 0 and red_y == 0:
            if dim == 2:
                reduced_img = self.class_windows[:, red_x:-1*red_x]
            elif dim == 3:
                reduced_img = self.class_windows[:, :, red_x:-1*red_x]
            elif dim == 4:
                reduced_img = self.class_windows[:, :, :, red_x:-1*red_x]
            else:
                print('ERROR: Please input a 2D or 3D or 4D Array in ReduceSize')
        self.class_windows = reduced_img
        return reduced_img
    
    def laplacian(self):
        lenz, leny, lenx = np.shape(self.class_imges)
        result = np.copy(self.class_imges())
        for i in range(lenz):
            stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            result[i, :, :] = sp.ndimage.convolve(
                self.class_imges[i, :, :], stencil, mode='nearest')
            self.class_imges = result
        return result
    
if __name__ == '__main__':
    print('this is the main function')