# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:42:58 2020

@author: Oliver
"""
import glob
import numpy as np
import pydicom

########################## FIND AND SAVE RADIOPTHERPY FILE PATHS ##########################
def save_globFilePaths(path):
    allRTPaths = []
    allRTPaths = glob.glob(path, recursive=True)
    np.save("RT Simulation Paths", allRTPaths)

########################## FILTER RT FILE PATHS AND SAVE RT STRUCTURE PATHS ##########################
def save_RTSimulationPaths(path):
    structureFiles = []
    allRTPaths = np.load(path)
    for i in range(0,len(allRTPaths)):
        if(pydicom.dcmread(allRTPaths[i]).Modality == 'RTSTRUCT'):
            print(allRTPaths[i])
            structureFiles.append(allRTPaths[i])
    np.save("RT Simulation Structure File Paths", structureFiles)
    print(len(structureFiles))
########################## FILTER RT FILE PATHS AND SAVE CT PATHS ##########################  
def save_CTImagePaths(path):
    imagesFolders = []
    allRTPaths = np.load(path)
    for i in range(0,len(allRTPaths)):
        if(pydicom.dcmread(allRTPaths[i]).Modality == 'CT'):
            length = len(allRTPaths[i])
            directoryOfCTImage = allRTPaths[i][0:length-10]
            print(directoryOfCTImage)
            imagesFolders.append(directoryOfCTImage)
            
    np.save("RT Simulation CT Image Folder Paths", imagesFolders)
    print(len(imagesFolders))
    
    