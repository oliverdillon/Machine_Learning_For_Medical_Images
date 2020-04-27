# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:26:41 2020

@author: Oliver
"""
from dicompylercore import dicomparser
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import scipy.ndimage
import collections
import random
import decimal
import math

from dicompylercore import util
from numbers import Number

import Data_Dictionary
import Transformation
from time import gmtime
import time


########################## OUTPUT FILE PATH OF INDIVIDUAL PATIENT##############################
def getOutputPath(structureFile):
    slashcount=0
    output_path =""
    for i, c in enumerate(structureFile):
        if(c =='\\'or c =='/'):
            slashcount+=1
            
        if(slashcount== 2):
            output_path =structureFile[:i+2]
    return output_path

def getImagePath(structureFile):
    slashcount=0
    output_path =""
    for i, c in enumerate(structureFile):
        if(c =='\\'or c =='/'):
            slashcount+=1
            
        if(slashcount== 3):
            output_path =structureFile[:i+2]
    return output_path
########################## LOAD FILES FOR PREPROCESSING ##############################
def load_Saved_Data(pathIndex,Organ,structureFiles,PrintInfo=False):
    ##Load
    output_path = getOutputPath(structureFiles[pathIndex])  
    imagesFolders = np.load('RT Simulation CT Image Folder Paths.npy')
    
    #Added this line so that the correct contour is loaded
    k = pathIndex
    print(getImagePath(structureFiles[pathIndex]))
    print(getImagePath(imagesFolders[k]))
    while(getImagePath(structureFiles[pathIndex]) !=  getImagePath(imagesFolders[k])):
        k +=1 #match the right contour with the right image
    
    ##CT Slices
    DicomImageSet = os.listdir(imagesFolders[k])
    
    #Output Checks
    if(PrintInfo): print("\n"+output_path)
    #print(structureFiles[pathIndex])
    #print(imagesFolders[k])
    Organ_Data = []
    ##Load Parotids Contours
    if (Organ.find("Right_Parotid")!=-1):
        Organ_Data = np.load(output_path+'Right_Contour_Parotids.npy', allow_pickle=True)
    elif (Organ.find("Left_Parotid")!=-1):
        Organ_Data = np.load(output_path+'Left_Contour_Parotids.npy', allow_pickle=True)
    elif (Organ.find("Brainstem")!=-1):
        Organ_Data = np.load(output_path+'Brainstem_Contour.npy', allow_pickle=True)
    elif (Organ.find("Right_Cochlea")!=-1):
        Organ_Data = np.load(output_path+'Right_Contour_Cochleas.npy', allow_pickle=True)
    elif (Organ.find("Left_Cochlea")!=-1):
        Organ_Data = np.load(output_path+'Left_Contour_Cochleas.npy', allow_pickle=True)
    return imagesFolders,k,DicomImageSet,Organ_Data

def get_First_Slice_Height(pathIndex,structureFiles):
    imagesFolders_Brainstem,imageFolderIndex_Brainstem,DicomImageSet_Brainstem,Brainstem_Data = load_Saved_Data(pathIndex,"Brainstem",structureFiles)
    zValues = []
    maxValue = -100000
    for z in range(0,len(DicomImageSet_Brainstem)-1):  
       ds_Brainstem = pydicom.read_file(imagesFolders_Brainstem[imageFolderIndex_Brainstem]+DicomImageSet_Brainstem[z])
       for i in range(0,len(Brainstem_Data)-1): 
            for j in range(0,len(Brainstem_Data[i])-1):
                num = str(ds_Brainstem.SliceLocation)
                num = decimal.Decimal(num)
                num = abs(num.as_tuple().exponent)
                loc = round(Brainstem_Data[i][j][2],num) 
                
                if (loc == ds_Brainstem.SliceLocation or Brainstem_Data[i][j][2] == ds_Brainstem.ImagePositionPatient[2]):
                    zValues.append(ds_Brainstem.SliceLocation)
                    if (Brainstem_Data[i][j][2]>maxValue):
                        maxValue =Brainstem_Data[i][j][2]
    return maxValue,zValues
########################## MANUAL EXTRACTION ##############################
def GetDefaultImageWindowLevel(dcm,rescaled_image,intercept,slope, window=0,level=0):     
    
    
    if ('WindowWidth' in dcm.ds) and ('WindowCenter' in dcm.ds):
        if isinstance(dcm.ds.WindowWidth, float):
            window = dcm.ds.WindowWidth
        elif isinstance(dcm.ds.WindowWidth, str):
            try:
                window = dcm.ds.WindowWidth
            except:
                print("image conversion error")
        elif isinstance(dcm.ds.WindowWidth, list):
            if (len(dcm.ds.WindowWidth) > 1):
                window = dcm.ds.WindowWidth[1]
        
        if isinstance(dcm.ds.WindowCenter, float):
            level = dcm.ds.WindowCenter
        elif isinstance(dcm.ds.WindowCenter, str):
            try:
                level = dcm.ds.WindowCenter
            except:
                print("image conversion error")
        elif isinstance(dcm.ds.WindowCenter, list):
            if (len(dcm.ds.WindowCenter) > 1):
                level = dcm.ds.WindowCenter[1]           
    
    if ((window, level) == (0, 0)):
        wmax = 0
        wmin = 0
        if (rescaled_image.max() > wmax):
            wmax = rescaled_image.max()
        if (rescaled_image.min() < wmin):
            wmin = rescaled_image.min()
        # Default window is the range of the data array
        window = int(wmax - wmin)
        # Default level is the range midpoint minus the window minimum
        level = int(window / 2 - abs(wmin))
    return window, level

def GetLUTValue(data, window, level):
        """Apply the RGB Look-Up Table for the data and window/level value."""

        lutvalue = util.piecewise(data,
                                [data <= (level - 0.5 - (window - 1) / 2),
                                 data > (level - 0.5 + (window - 1) / 2)],
                                [0, 255, lambda data:
                                 ((data - (level - 0.5)) / (window-1) + 0.5) *
                                 (255 - 0)])
        # Convert the resultant array to an unsigned 8-bit array to create
        # an 8-bit grayscale LUT since the range is only from 0 to 255
        return np.array(lutvalue, dtype=np.uint8)

def manualextration(ds,dcm,pixel_array,window,level):
        #rescale
    intercept, slope = 0, 1
    if ('RescaleIntercept' in ds and 'RescaleSlope' in ds):
        intercept = ds.RescaleIntercept if \
            isinstance(ds.RescaleIntercept, Number) else 0
        slope = ds.RescaleSlope if \
            isinstance(ds.RescaleSlope, Number) else 1       
    rescaled_image = pixel_array * slope + intercept
    
    #finish
    if(window == 0 and level == 0 ):
        window, level = GetDefaultImageWindowLevel(dcm, rescaled_image,intercept,slope)
    image = GetLUTValue(rescaled_image, window, level) ##converted to uint8
    #im = Image.fromarray(image).convert('L')
    
    return image


########################## OBTAIN CONTOURED CT IMAGES ##############################
def get_contoured_organ(pathIndex,Organ,key_Dict,no_Classes,structureFiles,flip = 0):
    ##Load Data
    imagesFolders,imageFolderIndex,DicomImageSet,Organ_Data = load_Saved_Data(pathIndex,Organ,structureFiles)
    TotalImageDictionary = {}
    maxValue, zValues = get_First_Slice_Height(pathIndex,structureFiles)
    DicomPatient,thickness = get_Patient(pathIndex,Organ,structureFiles)

    #Dimensions of Image
    imageDimensions = np.load("D:\HNSCC/ImageDimensions.npy", allow_pickle=True)
    #set orgin
    ds = pydicom.read_file(imagesFolders[imageFolderIndex]+DicomImageSet[0])

    #Cropping Image
    if(key_Dict.find("Uncropped")!=-1):
        height = 512 
        width = 512 
    else:
        padding =0
        height = imageDimensions[3]- imageDimensions[2]
        width = imageDimensions[1]- imageDimensions[0]
     
    if(Organ.find("Shift")!=-1):  
        x_Translation= random.randint(5,20)
        x_Translation= x_Translation*(-1)**random.randint(1,2)
        
        y_Translation= random.randint(5,20)
        y_Translation= y_Translation*(-1)**random.randint(1,2)
        
        zlowlimit = int(10/thickness)
        zhighlimit = int(30/thickness)
        zshift =random.randint(zlowlimit,zhighlimit)
        zshift= zshift*(-1)**random.randint(1,2)
        zshift *= thickness
    else:
        zshift = 0
    if(Organ.find("Bright")!=-1):
        xbright =random.randint(0,width)
        ybright =random.randint(0,height)
    isOrgan = False    
    #Image manipulation   
    for z in range(0,len(DicomImageSet)-1):        
        #Get CT Image and info
        ds = pydicom.read_file(imagesFolders[imageFolderIndex]+DicomImageSet[z])
        dcm = dicomparser.DicomParser(imagesFolders[imageFolderIndex]+DicomImageSet[z])
        #Only uses CT Images below the max height of the brainstem
        if(ds.ImagePositionPatient[2]<maxValue):

            #Create CT Image in the right format
            if(Organ.find("Bright")!=-1):
                ImageArray = ds.pixel_array
            elif(key_Dict.find("RGB")!=-1):
                image = dcm.GetImage(300,40)
                image = image.convert(mode ='RGB')
                ImageArray = np.asarray(image)
            else:
                if(key_Dict.find("Not_Windowed")!=-1):
                    image = dcm.GetImage()
                else:
                    image = dcm.GetImage(300,40)
                image = image.convert(mode ='L')
                ImageArray = np.asarray(image, 'uint8')
            
            #Create an array that contours can be mapped onto
            ImageArray=ImageArray.copy()
            ImageArray.flags.writeable = 1
            
            ContourPoints = []
            isOrgan = False #checks that the ct set has a contoured organ
            
            #Loops through contour data
            for i in range(0,len(Organ_Data)-1): 
                for j in range(0,len(Organ_Data[i])-1):
                    if(flip == 1):
                        pos =Organ_Data[i][j][2]
                        loc = ds.ImagePositionPatient[2]
                    else:
                        #accounts for rounding differences
                        num = str(ds.SliceLocation)
                        num = decimal.Decimal(num)
                        num = abs(num.as_tuple().exponent)
                        pos = round(Organ_Data[i][j][2],num) 
                        loc = ds.SliceLocation
                    
                    if (pos == loc+ zshift):
                        #maxValue-loc == maxValue-ds.SliceLocation+ zshift or
                        #Contouring
                        isOrgan =True
                        OrganIndex1 = int((Organ_Data[i][j][0]-ds.ImagePositionPatient[0])/ds.PixelSpacing[0])
                        OrganIndex2 = int((Organ_Data[i][j][1]-ds.ImagePositionPatient[1])/ds.PixelSpacing[1])
                        if(Organ.find("Bright")!=-1):
                            for k in range (xbright-10,xbright+10):
                                for f in range (ybright-10,ybright+10):
                                    ImageArray[k][f] =5000
                        
                        #Organ Contouring
                        if(key_Dict.find("RGB")!=-1):
                            #For RGB consideration:
                            if (Organ=="Left_Parotid"):
                                #make the left parotid show as blue 
                                ImageArray[OrganIndex2][OrganIndex1][2] = 255 
                            if (Organ=="Right_Parotid"): 
                                # make the right parotid show as green
                                ImageArray[OrganIndex2][OrganIndex1][1] = 255
                            if (Organ=="Brainstem"):
                                #make the brainstem show as red 
                                ImageArray[OrganIndex2][OrganIndex1][0] = 255 
                        else:
                            #For contour mask consideration:
                            ContourPoints.append((OrganIndex1,OrganIndex2))  
                    
            if(Organ.find("Bright")!=-1):
                if(Organ.find("Windowed")!=-1):
                    window =300
                    level =40
                else:
                    window= 0
                    level =0
                ImageArray =manualextration(ds,dcm,ImageArray,window,level)
                    
            if(key_Dict.find("RGB")!=-1):
                #For RGB consideration:
                if(key_Dict.find("Uncropped")!=-1):
                    Channel_Array = ImageArray
                else:
                    Channel_Array = ImageArray[imageDimensions[0]-padding: imageDimensions[1]+padding, 
                                                imageDimensions[2]-padding:imageDimensions[3]+padding,:]
            else:
                #For contour consideration:
                if(isOrgan ==True):
                    if(Organ.find("Aug")!=-1):
                        #For contour consistency checks
                        ContourPoints = Transformation.rotate_around_point(ContourPoints)
                        ContourPoints = Transformation.translate_points(ContourPoints)
                        ContourPoints = Transformation.shear_points(ContourPoints)
                        ContourPoints = Transformation.resize_points(ContourPoints)
                    if(Organ.find("Shift")!=-1):  
                        ContourPoints = Transformation.translate_contour(ContourPoints,x_Translation,y_Translation)
                    filled_Contour = Transformation.FillContourArea(ContourPoints)
                else:
                    filled_Contour = np.zeros((512,512), 'uint8')
                
                if(key_Dict.find("Uncropped")==-1):
                    ImageArray = ImageArray[imageDimensions[0]: imageDimensions[1], imageDimensions[2]:imageDimensions[3]]
                    filled_Contour = filled_Contour[ imageDimensions[0]: imageDimensions[1], imageDimensions[2]:imageDimensions[3]]
                
                
                if(key_Dict.find("Mask")!=-1):
                    Channel_Array = np.zeros((width,height,1), 'uint8')
                    Channel_Array = filled_Contour
                else:
                    #create a two channel image
                    Channel_Array = np.zeros((width,height,2), 'uint8')
                    Channel_Array[..., 0] = np.array(ImageArray)
                    Channel_Array[..., 1] = filled_Contour
                
            
            #Label Array so that the relevant organs can be filtered
            if(isOrgan ==True):
                TotalImageDictionary[ds.ImagePositionPatient[2]] = [Channel_Array,1]
            else:
                TotalImageDictionary[ds.ImagePositionPatient[2]] = [Channel_Array,0]
    
    OrderedImagesArray,label = sort_Data(TotalImageDictionary,Organ,key_Dict,no_Classes)
          
    if(OrderedImagesArray ==["False"] and flip == 0):
          OrderedImagesArray,label =get_contoured_organ(pathIndex,Organ,key_Dict,no_Classes,structureFiles,1)  
    
    return OrderedImagesArray,label

########################## SORT CT IMAGES ##############################
def sort_Data(TotalImageDictionary,Organ,key_Dict,no_Classes):
    
    #Correct the order of the array
    OrderedImagesDictionary = collections.OrderedDict(sorted(TotalImageDictionary.items()))
    OrderedImagesArray = []
    
    items = OrderedImagesDictionary.items()
    keys = [key for key, other in items]
    check = False
    
    
    if(key_Dict.find("3D")!=-1):
        if(key_Dict .find("Uncropped")!=-1):
            physicalDepth = 174
        else:
            physicalDepth = 135
        
        
        if(len(keys)!=0):
            top = keys[-1]
            #Extract only the relevant slices 
            for key, value in OrderedImagesDictionary.items():
                if(abs(top-key)<physicalDepth+1):
                    ImageArray =value[0] 
                    OrderedImagesArray.append(ImageArray) 
                    if (value[1] == 1):
                        #print(key)
                        check=True
        requiredNoHigh = math.floor(physicalDepth/2.5)
        requiredNo = physicalDepth/3
        print("Number of slices:%2i"%len(OrderedImagesArray))
        
        while(len(OrderedImagesArray)>requiredNoHigh):
            OrderedImagesArray.pop()
        while(len(OrderedImagesArray) > requiredNo and len(OrderedImagesArray) !=requiredNoHigh):
            OrderedImagesArray.pop()
            
        print("Required:%2i"%requiredNo)
    else:
        if(len(keys)!=0):
            top = keys[-1]
            #Extract only the relevant slices 
            for key, value in OrderedImagesDictionary.items():
                if (value[1] == 1):
                    ImageArray =value[0] 
                    OrderedImagesArray.append(ImageArray) 
                    check=True
                    
        #Get the middle of the organ
        numberOfSlices = len(OrderedImagesArray)
        middleOrganIndex = round(numberOfSlices/2)
    
        #Makes sure outputs are homogenous in dimensions
        if (key_Dict.find("11")!=-1):
            top=middleOrganIndex+6
            bottom =middleOrganIndex-5
        else:
            top=middleOrganIndex+2
            bottom =middleOrganIndex-1
        requiredNo = top-bottom
        OrderedImagesArray = OrderedImagesArray[bottom:top]

    #Automate Labelling process
    if(key_Dict.find("Aug")!=-1):
        Organs = ["Right_Parotid","Right_Parotid_Shift","Right_Parotid_Aug"]
    elif(key_Dict.find("Bright")!=-1):
        Organs = ["Right_Parotid_Bright","Null","Null"]
    else:
        Organs = ["Right_Parotid","Left_Parotid","Brainstem","Right_Cochlea","Left_Cochlea"]
    label = []
    
    for i in range(no_Classes):
        label.append(0)
        if (Organ==Organs[i]):
            label[i] =1
    
    label = np.array(label)
    
    numberOfSlices = len(OrderedImagesArray) 
    if(numberOfSlices<requiredNo):
        #If Slice not contoured
        print("Not enough Slices")
        return ["False"],"False"
    elif(check==False):
        print("No Contoured slices")
        return ["False"],"False"
    else:
        print(len(OrderedImagesArray))
        return np.array(OrderedImagesArray,'uint8'),label
################################# SAVE PREPROCESSED DATA #################################
def saveArray_2d(filename, arraySaveData,organ_count, key_Dict):
    X = []
    y = []
    
    print(np.array(arraySaveData).shape) 
    
    for features, label in arraySaveData:
        X.append(features)
        y.append(label)
    
    #Dimensions of Image
    imageDimensions = np.load("D:\HNSCC/ImageDimensions.npy", allow_pickle=True)
    
    TrainingFeaturesDict,TrainingLabelsDict = Data_Dictionary.get_Training_Dictionary()
    TestingFeaturesDict,TestingLabelsDict = Data_Dictionary.get_Testing_Dictionary()
    
    #Crop images
    if(key_Dict.find("Uncropped")!=-1):
        height = 512
        width = 512 
    else:
        height = imageDimensions[3]- imageDimensions[2]
        width = imageDimensions[1]- imageDimensions[0]
    
    
    if(key_Dict.find("RGB")!=-1):
        XNew = np.array(X).reshape(-1,width,height,3)
    else:
        XNew = np.array(X).reshape(-1,width,height,2)
        
    current_Time = time.strftime("%a, %d %b %Y %I:%M:%S %p %Z",time.gmtime())
    
    #Save Data
    if(filename == "Training"):
        stringTrainingDirectory = "D:/Image Preprocessing/Training"
        f = open(stringTrainingDirectory+"/TrainingCount.txt", "a")
        directory_Features = TrainingFeaturesDict[key_Dict]
        directory_Labels = TrainingLabelsDict[key_Dict]
    if(filename == "Testing"):
        stringTestingDirectory = "D:/Image Preprocessing/Testing"
        f = open(stringTestingDirectory+"/TestingCount.txt", "a")
        directory_Features = TestingFeaturesDict[key_Dict]
        directory_Labels = TestingLabelsDict[key_Dict]
    f.write(current_Time+", "+key_Dict+", "+organ_count+"\n")
    f.close()
        
    np.save(directory_Features, np.array(XNew))
    np.savetxt(directory_Labels, np.array(y), delimiter=",")
        
def saveArray_3d(filename, X,y,Organ, key_Dict,Patient_Name):    
    #Dimensions of Image
    imageDimensions = np.load("D:\HNSCC/ImageDimensions.npy", allow_pickle=True)
    
    TrainingFeaturesDict,TrainingLabelsDict = Data_Dictionary.get_Training_Dictionary()
    TestingFeaturesDict,TestingLabelsDict = Data_Dictionary.get_Testing_Dictionary()
    
    #Crop images
    if(key_Dict.find("Uncropped")!=-1):
        height = 512
        width = 512  
        depth =58
    else:
        height = imageDimensions[3]- imageDimensions[2]
        width = imageDimensions[1]- imageDimensions[0]
        depth= 45

    
    if(key_Dict.find("RGB")!=-1):
        XNew = np.array(X).reshape(width,height,depth,3)
    else:
        XNew = np.array(X).reshape(width,height,depth,2)
    
            
    print(np.array(X).shape)    
    #Save Data
    if(filename == "Training"):
        directory_Features = TrainingFeaturesDict[key_Dict]+Patient_Name+"_"+Organ
        directory_Labels = TrainingLabelsDict[key_Dict]+Patient_Name+"_"+Organ+".csv"
    if(filename == "Testing"):
        directory_Features = TestingFeaturesDict[key_Dict]+Patient_Name+"_"+Organ
        directory_Labels = TestingLabelsDict[key_Dict]+Patient_Name+"_"+Organ+".csv"
        
    np.save(directory_Features, np.array(XNew))
    np.savetxt(directory_Labels, np.array(y), delimiter=",")
        
    return directory_Features,directory_Labels
################################# INTERPOLATE DATA #################################
def get_Patient(pathIndex,Organ,structureFiles,PrintInfo=False):
    imagesFolders,imageFolderIndex,DicomImageSet,Organ_Data = load_Saved_Data(pathIndex,Organ,structureFiles,PrintInfo)
    
    images_in_Folder = [imagesFolders[imageFolderIndex] + image for image in DicomImageSet]

    slices = [pydicom.read_file(image) for image in images_in_Folder]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    
    
    #Get CT Image and info
    DicomPatient = slices[0]
    #DicomPatient = pydicom.read_file(imagesFolders[imageFolderIndex]+DicomImageSet[z])
    #DicomParserPatient = dicomparser.DicomParser(imagesFolders[imageFolderIndex]+DicomImageSet[z])
    thicknesses = []
    for i in range(len(slices)-1):
        thicknesses.append( slices[1].ImagePositionPatient[2] -slices[0].ImagePositionPatient[2])
    thickness = np.mean(thicknesses)
    if(PrintInfo): print("Thickness 1: %3.3f"%thickness)
    if(slices[0].SliceThickness!=""):
        print("Thickness 2: %3.3f"%slices[0].SliceThickness)
        
        if (thickness+0.25> slices[0].SliceThickness and thickness-0.25<slices[0].SliceThickness):
            thickness = slices[0].SliceThickness
    if(thickness%0.25):
        if(PrintInfo): print("Thickness 3: %3.3f"%thickness)
        #rounding method
        temp = thickness
        count = 0
        while(temp>0):
            temp-=0.25
            count+=1
        check = count*0.25
        if(PrintInfo): print("check 3: %3.3f"%check)
        if(thickness-check>0.125):
            thickness =check+0.25
        elif(thickness-check<-0.125):
            thickness =check-0.25
        else:
            thickness =check
                
            
        if(PrintInfo): print("Thickness 3: %3.3f"%thickness)
        if(thickness%0.25):
            thickness = math.trunc(thickness)
            if(PrintInfo): print("Thickness 4: %3.3f"%thickness)
    return DicomPatient,thickness
def interpolateArray(pathIndex,Organ,key_Dict,no_Classes,structureFiles):
   
    try:
        DicomPatient,thickness =get_Patient(pathIndex,Organ,structureFiles)
    except:
        print("Error when obtaining patient")
    tempArray, tempLabel =get_contoured_organ(pathIndex,Organ,key_Dict,no_Classes,structureFiles)
    if (tempLabel!="False"):
        tempArray = np.array(tempArray, "uint8")
        
        """
        # Determine current pixel spacing
        new_spacing=[1,1,1]
        spacing_list = [thickness]
        spacing_list.extend(DicomPatient.PixelSpacing)
        spacing = np.array(spacing_list, dtype=np.float32)
    
        #Calculate resize factor for interpolation
        resize_factor = spacing / new_spacing
        new_real_shape = tempArray[...,0].shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / tempArray[...,0].shape
        new_spacing = spacing / real_resize_factor
        """
        if(key_Dict.find("Mask")!=-1):
            outputArray= scipy.ndimage.interpolation.zoom(tempArray,(thickness/3,1,1) , order=0, mode='nearest')
        else:
            #Interpolation of the medical image real_resize_factor
            newArray = []
            newArray = scipy.ndimage.interpolation.zoom(tempArray[...,0],(thickness/3,1,1) , order=0, mode='nearest')
            shape = [s for s in np.array(newArray).shape]
            shape.append(2)
    
            outputArray = np.zeros(shape, "uint8")
            outputArray[...,0] = newArray
            
            newArray = scipy.ndimage.interpolation.zoom(tempArray[...,1], (thickness/3,1,1), order=0, mode='nearest')
            outputArray[...,1] = newArray
        
        return np.array(outputArray, "uint8"),tempLabel
    else:
        return tempArray, tempLabel
########################## IMAGE PREPROCESSING ##########################
def image_preprocessing_2d(start, end,key_Dict,no_Classes,structureFiles):
    #Initialise arrays
    Organs = ["Right_Parotid","Left_Parotid","Brainstem","Right_Cochlea","Left_Cochlea"] 
    neuralNetArray = []
    countDict = {}
    for organ in Organs:
            countDict[organ] = 0
    
    #Call Preprocessing Functions
    for pathIndex in range(start,end):
        for j in range(no_Classes):
            tempArray, tempLabel =get_contoured_organ(pathIndex,Organs[j],key_Dict,no_Classes,structureFiles)
            if(tempLabel!="False" and len(tempArray) ==100):
                for i in range(0,len(tempArray)):
                    neuralNetArray.append([tempArray[i],tempLabel])
                    countDict[Organs[j]] +=1
                print(np.array(tempArray).shape) 
                
    print("Pre Shuffle")   
    random.shuffle (neuralNetArray)
    print("Processing Complete")
    
    organ_count =""
    for organ in Organs:
        organ_count += organ+": %2i, "%countDict[organ]    
    print(organ_count)
    
    return neuralNetArray,organ_count
def image_preprocessing_3d(filename, start, end,key_Dict,no_Classes,structureFiles):  
    #Initialise arrays
    if(key_Dict.find("Aug")!=-1):
        Organs = ["Right_Parotid","Right_Parotid_Shift","Right_Parotid_Aug"]
    elif(key_Dict.find("Bright")!=-1):
        Organs = ["Right_Parotid_Bright_Windowed","Right_Parotid_Bright","NULL"]
    else:
        Organs = ["Right_Parotid","Left_Parotid","Brainstem","Right_Cochlea","Left_Cochlea"]
        
    countDict = {}
    directories =[]
    for organ in Organs:
            countDict[organ] = 0
    if(key_Dict.find("Uncropped")!=-1):
        requiredNo = 58
    else:
        requiredNo = 45
    #Call Preprocessing Functions
    for pathIndex in range(start,end):
        for j in range(no_Classes):
            DicomPatient,thickness = get_Patient(pathIndex,Organs[j],structureFiles,True)
            print(thickness)
            if(thickness== 3):
                tempArray, tempLabel =get_contoured_organ(pathIndex,Organs[j],key_Dict,no_Classes,structureFiles)
            else:
                tempArray, tempLabel =interpolateArray(pathIndex,Organs[j],key_Dict,no_Classes,structureFiles)
                
            if(tempLabel!="False"and len(tempArray) ==requiredNo):
                patient_name = structureFiles[pathIndex][9:22]
                directory_Features,directory_Labels = saveArray_3d( filename,tempArray,tempLabel,Organs[j],key_Dict,patient_name)
                directories.append([directory_Features,directory_Labels])
                countDict[Organs[j]] +=1
            else:
                print("Skipped: Only has %2i slices"%len(tempArray))
                
    print("Pre Shuffle")   
    random.shuffle (directories)
    print("Processing Complete")
    
    organ_count =""
    for organ in Organs:
        organ_count += organ+": %2i, "%countDict[organ]    
    print(organ_count)
    
    
    TrainingFeaturesDict,TrainingLabelsDict = Data_Dictionary.get_Training_Dictionary()
    TestingFeaturesDict,TestingLabelsDict = Data_Dictionary.get_Testing_Dictionary()
    
    featuresDirect = []
    labelsDirect = []
    
    for feature, label in directories:
        featuresDirect.append(feature)
        labelsDirect.append(label)
        
    current_Time = time.strftime("%a, %d %b %Y %I:%M:%S %p %Z",time.gmtime())
    if(filename=="Training"):
        stringTrainingDirectory = "D:/Image Preprocessing/Training"
        f = open(stringTrainingDirectory+"/TrainingCount.txt", "a")
        
        np.save(TrainingFeaturesDict[key_Dict]+"Patient Directories",featuresDirect)
        np.save(TrainingLabelsDict[key_Dict]+"Patient Directories",labelsDirect)
    elif(filename=="Testing"):
        stringTestingDirectory = "D:/Image Preprocessing/Testing"
        f = open(stringTestingDirectory+"/TestingCount.txt", "a")
        
        np.save(TestingFeaturesDict[key_Dict]+"Patient Directories",featuresDirect)
        np.save(TestingLabelsDict[key_Dict]+"Patient Directories",labelsDirect)
    f.write(current_Time+", "+key_Dict+", "+organ_count+"\n")
    f.close()

    print("SUCCESSS!")
    
########################## OBTAIN PAROTID CT IMAGES #############################

def isolate_Brainstem_Images(structureFiles,pathIndex,k):
    ds = pydicom.dcmread(structureFiles[pathIndex])
    imagesFolder = np.load('RT Simulation CT Image Folder Paths.npy')
    arrayFiles = os.listdir(imagesFolder[k]) 

    #Finds correct contour id
    referenceNumbersFound =[]

    for i in range(0,len(ds.StructureSetROISequence)):
        #Gets name of Structure
        contourName = ds.StructureSetROISequence[i].ROIName
        contourName = contourName.lower()

        if((contourName.find("brainstem")!=-1 or contourName.find("brain stem")!=-1 )and contourName.find("ex")==-1 
                    and contourName.find("2")==-1 and contourName.find("cm")==-1 and contourName.find("mm")==-1 and contourName.find("pv")==-1):
            #parotids references structure
            referenceNumbersFound.append(ds.StructureSetROISequence[i].ROINumber) 


    #Extracts Parotid CT scans from data set according to found contour ID
    ParotidCTImageFiles = []
    
    for i in range(0,len(ds.ROIContourSequence)):
        for referenceNumber in referenceNumbersFound:

            if (ds.ROIContourSequence[i].ReferencedROINumber == referenceNumber):
                contourSequences = ds.ROIContourSequence[i].ContourSequence

                for contourSequence in contourSequences:
                    contourslice = contourSequence.ContourImageSequence[0].ReferencedSOPInstanceUID

                    #Search through image files  
                    for z in range(0,len(arrayFiles)):
                            sliceDataset = pydicom.dcmread(imagesFolder[k]+arrayFiles[z])

                            if (sliceDataset.SOPInstanceUID ==contourslice):
                                
                                ParotidCTImageFiles.append(arrayFiles[z])
    
    return list(ParotidCTImageFiles)
########################## DETECT IMAGE SIZE FOR PREPROCESSING ##############################
def save_Image_Dimensions(structureFiles,Organs =["Brainstem"]):
    #Load files
    for Organ in Organs:
        print(Organ)
        for pathIndex in range(0,len(structureFiles)): 
            #Set maximal/minimal values
            minimumExternalIndex1 = 10000000
            maximumExternalIndex1 = -10000000
            minimumExternalIndex2 = 10000000
            maximumExternalIndex2 = -10000000
            
            #Load in Files:
            output_path = getOutputPath(structureFiles[pathIndex])
            imagesFolders,imageFolderIndex,DicomImageSet,Organ_Data = load_Saved_Data(pathIndex,Organ,structureFiles)  
            ds = pydicom.read_file(imagesFolders[imageFolderIndex]+DicomImageSet[0])
            
            Organ_Data = np.load(output_path+"External_Boundary_Contour.npy", allow_pickle=True)
            if(len(Organ_Data) == 0):
                print("Loading Ring Contour")
                Organ_Data = np.load(output_path+"Extended_Ring_Boundary_Contour.npy", allow_pickle=True)

            #Center_Data = np.load(output_path+"Isocenter_Contour.npy", allow_pickle=True)
            try:
                DicomImageSet = isolate_Brainstem_Images(structureFiles,pathIndex,imageFolderIndex)
                
                #Search through found datasets
                for z in range(0,len(DicomImageSet)-1):
                    ds = pydicom.read_file(imagesFolders[imageFolderIndex]+DicomImageSet[z])
                    dcm = dicomparser.DicomParser(imagesFolders[imageFolderIndex]+DicomImageSet[z])
                    
                    image = dcm.GetImage(300,40)
                    image = image.convert(mode ='RGB')
                    ImageArray = np.asarray(image)
                    ImageArray=ImageArray.copy()
                    ImageArray.flags.writeable = 1  
                    for i in range(0,len(Organ_Data)-1): 
                        for j in range(0,len(Organ_Data[i])-1):
                            
                            if (Organ_Data[i][j][2] == ds.ImagePositionPatient[2]):
                                #Get Relevant Coordinates
                                ExternalIndex2 = int((Organ_Data[i][j][0]-ds.ImagePositionPatient[0])/ds.PixelSpacing[0])
                                ExternalIndex1 = int((Organ_Data[i][j][1]-ds.ImagePositionPatient[1])/ds.PixelSpacing[1])
                                
                                #Update maximal/minimal values
                                if(ExternalIndex1<minimumExternalIndex1):
                                    minimumExternalIndex1 = ExternalIndex1
                                if(ExternalIndex1>maximumExternalIndex1):
                                    maximumExternalIndex1 = ExternalIndex1
                                    
                                if(ExternalIndex2<minimumExternalIndex2):
                                    minimumExternalIndex2 = ExternalIndex2
                                if(ExternalIndex2>maximumExternalIndex2):
                                    maximumExternalIndex2 = ExternalIndex2
           
            except:
                print("Error at index %2i"%pathIndex)                    
                
            #Save Values
            dimensions = [minimumExternalIndex1,maximumExternalIndex1,minimumExternalIndex2,maximumExternalIndex2]
            print("Centre Orgin: Left: %2.3f, Right: %2.3f, Bottom: %2.3f, Top: %2.3f"%(dimensions[0],dimensions[1],dimensions[2],dimensions[3]))
            np.save(output_path+"ImageDimensions_"+Organ, dimensions)  
            
    save_Final_Image_Dimensions(structureFiles,Organs)
def save_Final_Image_Dimensions(structureFiles,Organs):
    #Load files
    minimumExternalIndex1 = 10000000
    maximumExternalIndex1 = -10000000
    minimumExternalIndex2 = 10000000
    maximumExternalIndex2 = -10000000
    for Organ in Organs:
        for pathIndex in range(0,len(structureFiles)): 
            #Dimensions of Image
            output_path = getOutputPath(structureFiles[pathIndex])
            print(output_path)
            imageDimensions = np.load(output_path+"ImageDimensions_"+Organ+".npy", allow_pickle=True)
            
            if(imageDimensions[0]<minimumExternalIndex1):
                minimumExternalIndex1 = imageDimensions[0]
            if(imageDimensions[1]>maximumExternalIndex1):
                maximumExternalIndex1 = imageDimensions[1]
            if(imageDimensions[2]<minimumExternalIndex2):
                minimumExternalIndex2 = imageDimensions[2]
            if(imageDimensions[3]>maximumExternalIndex2):
                maximumExternalIndex2 = imageDimensions[3]
                
            print(imageDimensions)
        
    dimensions = [minimumExternalIndex1,maximumExternalIndex1,minimumExternalIndex2,maximumExternalIndex2] 
    print("Left: %2.3f, Right: %2.3f, Bottom: %2.3f, Top: %2.3f"%(dimensions[0],dimensions[1],dimensions[2],dimensions[3]))
    np.save("D:\HNSCC/ImageDimensions", dimensions)