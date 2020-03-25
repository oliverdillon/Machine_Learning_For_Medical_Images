# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:33:34 2020

@author: Oliver
"""
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import Image_Preprocessing
import tensorflow as tf

########################## OBTAIN DIFFERENT SAGITTAL AND CORONAL VIEW OF IMAGES  ##############################
def DICOM_3D_Plots(pixelSpacing,Thickness, allImages,Directory,fig_Name,index1=0,index2=0,index3=0):
    # pixel aspects, assuming all slices are the same
    ps = pixelSpacing
    ss = Thickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]

    # create 3D array
    img_shape = list(allImages.shape[1:3])
    img_shape.append(len(allImages))
    img3d=np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, img2d in enumerate(allImages):
        img3d[:,:,i] = img2d
    
    if(fig_Name.find("Contour") !=-1):
        #index3 = img_shape[2]-1
        #index2 = img_shape[1]-1
        #index1 = img_shape[0]-1
        while(np.sum(img3d[:,:,index3])==0):
            index3+=1
        while(np.sum(img3d[:,index2,:])==0):
            index2+=1
        while(np.sum(img3d[index1,:,:])==0):
            index1+=1
        index3+=4
        index2+=4
        index1+=4
        while(np.sum(img3d[:,:,index3])==0):
            index3-=1
        while(np.sum(img3d[:,index2,:])==0):
            index2-=1
        while(np.sum(img3d[index1,:,:])==0):
            index1-=1
     
    # plot 3 orthogonal slices
    a1 = plt.subplot(111)
    plt.imshow(img3d[:,:,index3],cmap="gray")
    a1.set_aspect(ax_aspect)
    plt.title("Axial Middle_"+fig_Name)
    plt.savefig(Directory+"Axial\\"+fig_Name+".png")
    plt.show()

    a2 = plt.subplot(111)
    plt.imshow(img3d[:,index2,:],cmap="gray")
    a2.set_aspect(sag_aspect)
    plt.title("Sagittal Middle_"+fig_Name)
    plt.savefig(Directory+"Sagittal\\"+fig_Name+".png")
    plt.show()

    a3 = plt.subplot(111)
    plt.imshow(img3d[index1,:,:].T,cmap="gray")
    a3.set_aspect(cor_aspect)
    plt.title("Coronal Middle_"+fig_Name)
    plt.savefig(Directory+"Coronal\\"+fig_Name+".png")
    plt.show()
    
    return index1,index2,index3

def Overlay_DICOM_3D_Plots(pixelSpacing,Thickness,AllCTImages,allContourImages,Directory,fig_Name):
    # pixel aspects, assuming all slices are the same
    ps = pixelSpacing
    ss = Thickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]

    # create 3D array
    img_shape = list(allContourImages.shape[1:3])
    img_shape.append(len(allContourImages))
    img3d_Contour=np.zeros(img_shape)
    img3d_CTImage=np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, img2d in enumerate(allContourImages):
        img3d_Contour[:,:,i] = img2d
        
    for i, img2d in enumerate(AllCTImages):
        img3d_CTImage[:,:,i] = img2d
    
    index1 = 0
    index2 = 0
    index3 = 0
    
    while(np.sum(img3d_Contour[:,:,index3])==0):
        index3+=1
    while(np.sum(img3d_Contour[:,index2,:])==0):
        index2+=1
    while(np.sum(img3d_Contour[index1,:,:])==0):
        index1+=1
    index3+=4
    index2+=4
    index1+=4
    while(np.sum(img3d_Contour[:,:,index3])==0):
        index3-=1
    while(np.sum(img3d_Contour[:,index2,:])==0):
        index2-=1
    while(np.sum(img3d_Contour[index1,:,:])==0):
        index1-=1
    
    if(fig_Name.find("Middle")!=-1):
        index1 =img3d_Contour.shape[0]//2
        index2 =img3d_Contour.shape[1]//2
        index3 =img3d_Contour.shape[2]//2
        
    overlayAxialshape  = list(img3d_Contour[:,:,index3].shape)
    overlayAxialshape.append(3)
    overlayAxial =np.zeros(overlayAxialshape,'uint8')
    
    overlayAxial[...,0] = img3d_CTImage[:,:,index3]
    overlayAxial[...,2] = img3d_CTImage[:,:,index3]
    overlayAxial[...,1] = img3d_Contour[:,:,index3]
    
    overlaySagittalshape  = list(img3d_Contour[:,index2,:].shape)
    overlaySagittalshape.append(3)
    overlaySagittal =np.zeros(overlaySagittalshape,'uint8')
    
    overlaySagittal[...,0] = img3d_CTImage[:,index2,:]
    overlaySagittal[...,2] = img3d_CTImage[:,index2,:]
    overlaySagittal[...,1] = img3d_Contour[:,index2,:]

    overlayCoronalshape  = list((img3d_Contour[index1,:,:].T).shape)
    overlayCoronalshape.append(3)
    overlayCoronal =np.zeros(overlayCoronalshape,'uint8')
    
    overlayCoronal[...,0] = img3d_CTImage[index1,:,:].T
    overlayCoronal[...,2] = img3d_CTImage[index1,:,:].T
    overlayCoronal[...,1] = img3d_Contour[index1,:,:].T
    
    print(Directory+"Axial\\"+fig_Name+".png")
    # plot 3 orthogonal slices
    a1 = plt.subplot(111)
    plt.imshow(overlayAxial)
    a1.set_aspect(ax_aspect)
    plt.title("Axial_"+fig_Name)
    plt.savefig(Directory+"Axial\\"+fig_Name+".png",dpi=480)
    plt.show()
    
    a2 = plt.subplot(111)
    plt.imshow(overlaySagittal)
    a2.set_aspect(sag_aspect)
    plt.title("Sagittal_"+fig_Name)
    plt.savefig(Directory+"Sagittal\\"+fig_Name+".png",dpi=480)
    plt.show()
    
    a3 = plt.subplot(111)
    plt.imshow(overlayCoronal)
    a3.set_aspect(cor_aspect)
    plt.title("Coronal_"+fig_Name)
    plt.savefig(Directory+"Coronal\\"+fig_Name+".png",dpi=480)
    plt.show()
    
########################## CT DATA DISPLAY ##############################  
def plot_CT_Image_Histograms(stringImageDirectory,structureFiles):
    stringImageDirectory ="C:/Users/Oliver/Documents/University/Year 4/MPhys Project/Images/"
    allheight =[]
    allThickness =[]
    allNoSlices =[]

    for pathIndex in range(len(structureFiles)):
        Organ = "Right_Parotid"

        imagesFolders,imageFolderIndex,DicomImageSet,Organ_Data = Image_Preprocessing.load_Saved_Data(pathIndex,Organ)
        images_in_Folder = [imagesFolders[imageFolderIndex] + image for image in DicomImageSet]
        slices = [pydicom.read_file(image) for image in images_in_Folder]
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        thickness = slices[0].SliceThickness

        if (thickness == ""):
            thickness =  slices[0].SliceLocation - slices[1].SliceLocation

        thic =np.abs(thickness)
        height = slices[0].SliceLocation - slices[-1].SliceLocation

        if (thic ==3):
            allheight.append(np.abs(height))
            allThickness.append(np.abs(thickness))
            allNoSlices.append(len(slices))

    plt.hist(allheight)
    plt.title("Height of Scans")
    plt.savefig(stringImageDirectory+"Height of Scans_Just 3mm.png")
    plt.show()

    plt.hist(allThickness)
    plt.title("Thickness of Scans")
    plt.savefig(stringImageDirectory+"Thickness of Scans_Just 3mm.png")
    plt.show()

    plt.hist(allNoSlices)
    plt.title("No of Slices")
    plt.savefig(stringImageDirectory+"No of Slices_Just 3mm.png")
    plt.show() 

################################# PLOT CROPPED GRAPHS #################################
def plot_Cropped_Graphs(Organ,key_Dict,no_Classes,structureFiles):  
    ###Initialise
    stringImageDirectory ="C:/Users/Oliver/Documents/University/Year 4/MPhys Project/Images/"
    
    #Right Parotid Plot
    Organ = "Right_Parotid"
    pathIndex = 0
    OrderedImagesArray,label = Image_Preprocessing.get_contoured_organ(pathIndex,Organ,key_Dict,no_Classes)
    plt.imshow(OrderedImagesArray[0])
    patient_name = structureFiles[pathIndex][9:22] 
    plt.axis("off")
    plt.savefig(stringImageDirectory+patient_name+"_"+Organ+"_"+key_Dict+".png")
    plt.show()
    
########################## CHECK DATA HOMOGENEITY ##########################
def check_Homogeneity(pathIndex,organ, directory,structureFiles):
    tempArray, tempLabel =Image_Preprocessing.get_contoured_organ(pathIndex,organ,512)
    index = 1
    if(tempLabel!="False"):
        fig,ax = plt.subplots(1,2,figsize=[6,3])

        ax[0].imshow(tempArray[index][..., 0], cmap='gray')
        ax[0].axis('off')
        ax[0].set_title('CT Slice')
        ax[1].imshow(tempArray[index][..., 1], cmap='gray')
        ax[1].axis('off')
        ax[1].set_title('Contour')

        patient_name = structureFiles[pathIndex][9:22]
        plt.savefig(directory+patient_name+".png")
        plt.show()
