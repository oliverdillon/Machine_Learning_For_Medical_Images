# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:51:07 2020

@author: Oliver
"""
from dicompylercore import dicomparser
import numpy as np

########################## OBTAIN AND SAVE PAROTID STRUCTURE CONTOURS ##############################
def save_Parotid_Contours(structureFiles):
    noOfRightParotid =0
    noOfLeftParotid =0
    for z in range(0,len(structureFiles)):
        RTParotid = []
        LTParotid = []
        output_path = structureFiles[z][0:23]
        ##Contours
        dataset = dicomparser.DicomParser(structureFiles[z])
        structures = dataset.GetStructures()
        for i in range(1,len(structures)): #to strip out the parotids
            try:
                StructureCoordDictsLeft =[]
                StructureCoordDictsRight =[]
                name = str(structures[i]['name']).lower()                 
    
                if (name.find("sub") ==-1 and name.find("total") ==-1 and name.find("def") ==-1 
                   and name.find("sup") ==-1 and name.find("deep") ==-1 and name.find("gy") ==-1 
                   and name.find("avoid") ==-1 and name.find("ptv") ==-1 and name.find("push") ==-1 and name.find("tail") ==-1): 
                    if (name.find("parotid")!=-1 or name.find("prtd")!=-1):
                        ##Right Parotid
                        if (name.find("rt")!=-1 or name.find("r ")!=-1 or name.find(" r")!=-1 or name.find("right")!=-1):
                            StructureCoordDictsRight.append(dataset.GetStructureCoordinates(i))
                            for j in list(StructureCoordDictsRight[0]): #iterate through the dictionary i get in line 14
                                        RTParotid.append(StructureCoordDictsRight[0][j][0]['data']) #pull out only the matrix of xyz values
                            noOfRightParotid +=1 
                            
                        ##Left Parotid           
                        if (name.find("lt")!=-1 or name.find("l ")!=-1 or name.find(" l")!=-1 or name.find("left")!=-1):
                            StructureCoordDictsLeft.append(dataset.GetStructureCoordinates(i))
                            for j in list(StructureCoordDictsLeft[0]): #iterate through the dictionary i get in line 14
                                        LTParotid.append(StructureCoordDictsLeft[0][j][0]['data']) #pull out only the matrix of xyz values
                            noOfLeftParotid +=1 
                        
                np.save(output_path+"Right_Contour_Parotids", RTParotid)#saves contours in correct folder
                np.save(output_path+"Left_Contour_Parotids", LTParotid)#saves contours in correct folder
            except:
                print("Error at Index %2i"%z) 
                #11,12,28,39, 42,62  Not Contoured but have pcs?
    
    print("Number of right parotid: %2i"%noOfRightParotid)
    print("Number of left parotid: %2i"%noOfLeftParotid)
    
########################## OBTAIN AND SAVE RING STRUCTURE CONTOURS ##############################
def save_Ring_Contours(structureFiles):
    noOfRings =0
    for z in range(0,len(structureFiles)):
        ring =[]
        output_path = structureFiles[z][0:23]
        ##Contours
        dataset = dicomparser.DicomParser(structureFiles[z])
        structures = dataset.GetStructures()
        for i in range(1,len(structures)): #to strip out the parotids
            try:
                StructureCoordDicts =[]
                name = str(structures[i]['name']).lower()
                if (name.find("ring")!=-1 and name.find("inner")==-1 and name.find("ext")==-1):
                    StructureCoordDicts.append(dataset.GetStructureCoordinates(i))
                    for j in list(StructureCoordDicts[0]): #iterate through the dictionary i get in line 14
                        ring.append(StructureCoordDicts[0][j][0]['data']) #pull out only the matrix of xyz value
                    noOfRings +=1
                np.save(output_path+"Extended_Ring_Boundary_Contour", ring)#saves contours in correct folder
            except:
                print("Error at Index %2i"%z)
                #123,128,136 
    print("Number of ring: %2i"%noOfRings)
    
########################## OBTAIN AND SAVE EXTERNAL STRUCTURE CONTOURS ##############################
def save_External_Contours(structureFiles):
    noOfExternal =0
    for z in range(0,len(structureFiles)):
        external =[]
        output_path = structureFiles[z][0:23]
        ##Contours
        dataset = dicomparser.DicomParser(structureFiles[z])
        structures = dataset.GetStructures()
        for i in range(1,len(structures)): #to strip out the parotids
            try:
                StructureCoordDicts =[]
                name = str(structures[i]['name']).lower()
                if (name.find("external")!=-1):
                    StructureCoordDicts.append(dataset.GetStructureCoordinates(i))
                    for j in list(StructureCoordDicts[0]): #iterate through the dictionary i get in line 14
                        external.append(StructureCoordDicts[0][j][0]['data']) #pull out only the matrix of xyz value
                    noOfExternal +=1 
                np.save(output_path+"External_Boundary_Contour", external)
            except:
                print("Error at Index %2i"%z)
                #123,128,136 
    print("Number of external: %2i"%noOfExternal)
########################## OBTAIN AND SAVE BRAIN STEM STRUCTURE CONTOURS ##############################
def save_Brainstem_Contours(structureFiles):
    noOfBrainstem =0
    for z in range(0,len(structureFiles)):
        Brainstem =[]
        output_path = structureFiles[z][0:23]
        ##Contours
        dataset = dicomparser.DicomParser(structureFiles[z])
        structures = dataset.GetStructures()
        for i in range(1,len(structures)): #to strip out the parotids
            try:
                StructureCoordDicts =[]
                name = str(structures[i]['name']).lower()
                if ((name.find("brainstem")!=-1 or name.find("brain stem")!=-1 )and name.find("ex")==-1 
                    and name.find("2")==-1 and name.find("cm")==-1 and name.find("mm")==-1 and name.find("pv")==-1):
                    StructureCoordDicts.append(dataset.GetStructureCoordinates(i))
                    for j in list(StructureCoordDicts[0]): #iterate through the dictionary i get in line 14
                        Brainstem.append(StructureCoordDicts[0][j][0]['data']) #pull out only the matrix of xyz value
                    noOfBrainstem += 1
                np.save(output_path+"Brainstem_Contour", Brainstem)#saves contours in correct folder
            except:
                print("Error at Index %2i"%z) 
    
    print("Number of Brainstem: %2i"%noOfBrainstem)
########################## OBTAIN AND SAVE ISOCENTRE STRUCTURE CONTOURS ##############################
def save_Isocentre_Contours(structureFiles):
    noOfIsocentres =0
    for z in range(0,len(structureFiles)):
        Isocentres =[]
        output_path = structureFiles[z][0:23]
        ##Contours
        dataset = dicomparser.DicomParser(structureFiles[z])
        structures = dataset.GetStructures()
        for i in range(1,len(structures)): #to strip out the parotids
            try:
                StructureCoordDicts =[]
                name = str(structures[i]['name']).lower()
                if ((name.find("iso")!=-1)and name.find("final")==-1):
                    StructureCoordDicts.append(dataset.GetStructureCoordinates(i))
                    for j in list(StructureCoordDicts[0]): #iterate through the dictionary i get in line 14
                        Isocentres.append(StructureCoordDicts[0][j][0]['data']) #pull out only the matrix of xyz value
                    noOfIsocentres += 1
                np.save(output_path+"Isocenter_Contour", Isocentres)#saves contours in correct folder
            except:
                print("Error at Index %2i"%z) 
    
    print("Number of Isocentres: %2i"%noOfIsocentres) 
########################## OBTAIN AND SAVE COCHLEA STRUCTURE CONTOURS ##############################
def save_Cochlea_Contours(structureFiles):
    noOfRightCochlea =0
    noOfLeftCochlea =0
    for z in range(0,len(structureFiles)):
        
        RTCochlea = []
        LTCochlea = []
        output_path = structureFiles[z][0:23]
        ##Contours
        dataset = dicomparser.DicomParser(structureFiles[z])
        structures = dataset.GetStructures()
    
        for i in range(1,len(structures)): #to strip out the parotids
            try:
                StructureCoordDictsLeft =[]
                StructureCoordDictsRight =[]
                name = str(structures[i]['name']).lower()                 
    
                if (name.find("sub") ==-1 and name.find("total") ==-1 and name.find("def") ==-1 
                   and name.find("sup") ==-1 and name.find("deep") ==-1 and name.find("gy") ==-1 
                   and name.find("avoid") ==-1 and name.find("ptv") ==-1 and name.find("push") ==-1 and name.find("tail") ==-1): 
                    if (name.find("cochlea")!=-1 or name.find("prtd")!=-1):
                        
                        ##Right Cochlea
                        if (name.find("rt")!=-1 or name.find("r ")!=-1 or name.find(" r")!=-1 or name.find("right")!=-1):
                            StructureCoordDictsRight.append(dataset.GetStructureCoordinates(i))
                            for j in list(StructureCoordDictsRight[0]): #iterate through the dictionary i get in line 14
                                        RTCochlea.append(StructureCoordDictsRight[0][j][0]['data']) #pull out only the matrix of xyz values
                            noOfRightCochlea +=1
                            
                        ##Left Cochlea          
                        if (name.find("lt")!=-1 or name.find("l ")!=-1 or name.find(" l")!=-1 or name.find("left")!=-1):
                            StructureCoordDictsLeft.append(dataset.GetStructureCoordinates(i))
                            for j in list(StructureCoordDictsLeft[0]): #iterate through the dictionary i get in line 14
                                        LTCochlea.append(StructureCoordDictsLeft[0][j][0]['data']) #pull out only the matrix of xyz values
                            noOfLeftCochlea +=1
    
                        
                np.save(output_path+"Right_Contour_Cochleas", RTCochlea)#saves contours in correct folder
                np.save(output_path+"Left_Contour_Cochleas", LTCochlea)#saves contours in correct folder
            except:
                print("Error at Index %2i"%z)
    print("Number of Right Cochlea: %2i"%noOfRightCochlea)
    print("Number of Left Cochlea: %2i"%noOfLeftCochlea)