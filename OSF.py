#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:48:01 2023

@author: janus
"""

import numpy as np
import sys

class ReadStar(object):
    '''
    Read in relion data STAR files
    '''
    def __init__(self, infile):
        self.infile = infile
        self._var = None
        self._data = None
        self._optics = None

    def var(self):
        '''
        Variables within the particle data loop of the relion STAR
        file
        '''
        if self._var:
            return self._var
        else:
            self._parse_file()
            return self._var

    def data(self):
        '''
        Data loop of the relion STAR file as a list
        '''
        if self._data:
            return self._data
        else:
            self._parse_file()
            return self._data

    def data_array(self, nested=False):
        '''
        Data loop of the relion STAR file as an array
        '''
        if self._data:
            if (len(self._var) == len(self._data[0])):
                return np.asarray(self._data)
            else:
                print("ERROR: Number of metadata variables does not match number of columns in the input STAR file")
                sys.exit(2)
        elif nested:
            print("ERROR: Recursion error when reading file")
            sys.exit(2)
        else:
            self._parse_file()
            return self.data_array(nested=True)

    def optics(self):
        '''
        New relion STAR file format includes optics table, 
        accessible by this method.
        '''
        if self._optics:
            return self._optics
        else:
            self._parse_file()
            return self._optics

    def _parse_file(self):
        var = [] #read the variables
        data = [] # read the data
        optics = [] # store new relion optics data table

        new_relion_star_format = False

        for line in open(self.infile).readlines():
            if line[0] == "#":
                continue
            elif "data_optics" in line: 
                new_relion_star_format = "OPTICS"
            elif "data_particles" in line:
                new_relion_star_format = "PARTICLES"

            if new_relion_star_format == "OPTICS":
                optics.append(line.strip())

            elif "_rln" in line:
                var.append(line.split()[0])
            elif ("data_" in line or
                    "loop_" in line or
                    len(line.strip()) == 0):
                continue
            else:
                data.append(line.split())
        self._var = var
        self._data = data
        self._optics = optics

def CHEP(infile):        
    instar = ReadStar(infile)
    rlnvar = instar.var()
    rlndata_list = instar.data()
    rlndata = instar.data_array()
    rlnoptics = instar.optics()
    
    rlnvar_nr = len(rlnvar)
    print("Number of parameters in the relion file = " + str(int(rlnvar_nr)))
    
    ptcl_nr=len(rlndata_list)
    print("Number of particles in the relion file = " + str(int(ptcl_nr)))
    
    rlntubeID_col = rlnvar.index('_rlnHelicalTubeID')
    rlnclassID_col = rlnvar.index('_rlnClassNumber')
    rlnMicrographName_col = rlnvar.index('_rlnMicrographName')
    
    #initialising array
    helixIDclass_data = np.zeros((ptcl_nr, 2))
    
    helixdict = {}
    counter = 1
    
    # renaming the helical_id
    for ptcl in range(ptcl_nr):
        uniqueID = (rlndata[ptcl, rlnMicrographName_col] + '-' + rlndata[ptcl,rlntubeID_col])
        #check whether ID exists in dictionary
        if uniqueID in helixdict:
            helixID = helixdict[uniqueID]
        else:
            #add unseen ID to the dictionary
            helixID = counter
            counter += 1
            helixdict[uniqueID] = helixID
        helixIDclass_data[ptcl,0] = np.int_(helixID)
    
    print("Number of Full-length helical proteins = " + str(int(helixID)))
    
    helix_list= np.zeros((ptcl_nr,1))
    print("Number of particles in the dataset: " + str(ptcl_nr))
    
    rlndata = np.append(rlndata, helixIDclass_data, axis=1)
    helixIDclass_data[:,1] = np.int_(rlndata[:,rlnclassID_col])
    rlnclassID_max = np.int_(rlndata[:,[rlnclassID_col]]).max(axis=0)
    rlnclassID_min = np.int_(rlndata[:,[rlnclassID_col]]).min(axis=0)
    helixID_max = np.int_(helixIDclass_data[:,0].max(axis=0))
    helixID_min = np.int_(helixIDclass_data[:,0].min(axis=0))
    
    print("Number of 2D classes = " + str(len(np.unique(helixIDclass_data[:,1]))))
    
    fc_matrix = np.zeros((int(helixID_max), int(rlnclassID_max)))
    
    for ptcl in range(ptcl_nr):
        f, c = helixIDclass_data[ptcl,0], helixIDclass_data[ptcl,1]
        fc_matrix[int(f-1), int(c-1)] +=1
    fc_matrix = fc_matrix / fc_matrix.sum(axis=1,keepdims=True)
    return fc_matrix
