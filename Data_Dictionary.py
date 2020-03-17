# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:45:04 2020

@author: Oliver
"""

def get_Testing_Dictionary():
    stringTestingDirectory = "D:/Image Preprocessing/Testing/Uncropped"
    stringTesting3DDirectory = "D:\Patient Preprocessing\Testing/Uncropped/"
    TestingFeaturesDict = {"3Class2Channel3Slice":stringTestingDirectory[:30]+"\Testing_Features_2D_3Class_2Channel_3Slice.npy",
                           "3ClassRGB3Slice":stringTestingDirectory[:30]+"\Testing_Features_2D_3Class_RGB_3Slice.npy",
                           "3Class2Channel3Slice_Uncropped":stringTestingDirectory+"\Testing_Features_2D_3Class_2Channel_3Slice.npy",
                           "3ClassRGB3Slice_Uncropped":stringTestingDirectory+"\Testing_Features_2D_3Class_RGB_3Slice.npy",
                           "Parotids11Slice_Uncropped":stringTestingDirectory+"\Testing_Features_2D_Parotids_2Channel_11Slice.npy",
                           "RightParotids11Slice_Aug":stringTestingDirectory[:30]+"\Testing_Features_2D_RightParotids_2Channel_11Slice_Aug.npy",
                           "2Class2Channel3D_Uncropped":stringTestingDirectory+"\Testing_Features_3D_2Class_2Channel.npy",
                           "3Class2Channel3D_Uncropped":stringTesting3DDirectory+"Features/",
                           "3Class2Channel3D":stringTesting3DDirectory[:33]+"Features/",
                           "3Class3Channel3Slice":stringTestingDirectory+"\Testing_Features_2D_3Class_3Channel_3Slice.npy",
                           "5Class2Channel3Slice":stringTestingDirectory+"\Testing_Features_2D_5Class_2Channel_3Slice.npy",
                           "5Class3Channel3Slice":stringTestingDirectory+"\Testing_Features_2D_5Class_3Channel_3Slice.npy",
                          }
    
    TestingLabelsDict = {"3Class2Channel3Slice":stringTestingDirectory[:30]+"\Testing_Label_2D_3Class_2Channel_3Slice.csv",
                         "3ClassRGB3Slice":stringTestingDirectory[:30]+"\Testing_Label_2D_3Class_RGB_3Slice.csv",
                         "3Class2Channel3Slice_Uncropped":stringTestingDirectory+"\Testing_Label_2D_3Class_2Channel_3Slice.csv",
                         "3ClassRGB3Slice_Uncropped":stringTestingDirectory+"\Testing_Label_2D_3Class_RGB_3Slice.csv",
                         "Parotids11Slice_Uncropped":stringTestingDirectory+"\Testing_Label_2D_Parotids_2Channel_11Slice.csv",
                         "RightParotids11Slice_Aug":stringTestingDirectory[:30]+"\Testing_Label_2D_RightParotids_2Channel_11Slice_Aug.csv",
                         "2Class2Channel3D_Uncropped":stringTestingDirectory+"\Testing_Label_3D_2Class_2Channel.csv",
                         "3Class2Channel3D_Uncropped":stringTesting3DDirectory+"Labels/",
                         "3Class2Channel3D":stringTesting3DDirectory[:33]+"Labels/",
                         "3Class3Channel3Slice":stringTestingDirectory+"\Testing_Label_2D_3Class_3Channel_3Slice.csv",
                         "5Class2Channel3Slice":stringTestingDirectory+"\Testing_Label_2D_5Class_2Channel_3Slice.csv",
                         "5Class3Channel3Slice":stringTestingDirectory+"\Testing_Label_2D_5Class_3Channel_3Slice.csv",
                          }
    return TestingFeaturesDict,TestingLabelsDict

def get_Training_Dictionary():
    stringTrainingDirectory = "D:/Image Preprocessing/Training/Uncropped"
    stringTraining3DDirectory = "D:\Patient Preprocessing\Training/Uncropped/"
    TrainingFeaturesDict = {"3Class2Channel3Slice":stringTrainingDirectory[:31]+"\Training_Features_2D_3Class_2Channel_3Slice.npy",
                            "3ClassRGB3Slice":stringTrainingDirectory[:31]+"\Training_Features_2D_3Class_RGB_3Slice.npy",
                            "3Class2Channel3Slice_Uncropped":stringTrainingDirectory+"\Training_Features_2D_3Class_2Channel_3Slice.npy",
                            "3ClassRGB3Slice_Uncropped":stringTrainingDirectory+"\Training_Features_2D_3Class_RGB_3Slice.npy",
                            "Parotids11Slice_Uncropped":stringTrainingDirectory+"\Training_Features_2D_Parotids_2Channel_11Slice.npy",
                            "RightParotids11Slice_Aug":stringTrainingDirectory[:31]+"\Training_Features_2D_RightParotids_2Channel_11Slice_Aug.npy",
                            "2Class2Channel3D_Uncropped":stringTrainingDirectory+"\Training_Features_3D_2Class_2Channel.npy",
                            "3Class2Channel3D_Uncropped":stringTraining3DDirectory+"Features/",
                            "3Class2Channel3D":stringTraining3DDirectory[:33]+"Features/",
                            "3Class3Channel3Slice":stringTrainingDirectory+"\Training_Features_2D_3Class_3Channel_3Slice.npy",
                            "5Class2Channel3Slice":stringTrainingDirectory+"\Training_Features_2D_5Class_2Channel_3Slice.npy",
                            "5Class3Channel3Slice":stringTrainingDirectory+"\Training_Features_2D_5Class_3Channel_3Slice.npy",
                          }
    TrainingLabelsDict = {"3Class2Channel3Slice":stringTrainingDirectory[:31]+"\Training_Label_2D_3Class_2Channel_3Slice.csv",
                          "3ClassRGB3Slice":stringTrainingDirectory[:31]+"\Training_Label_2D_3Class_RGB_3Slice.csv",
                          "3Class2Channel3Slice_Uncropped":stringTrainingDirectory+"\Training_Label_2D_3Class_2Channel_3Slice.csv",
                          "3ClassRGB3Slice_Uncropped":stringTrainingDirectory+"\Training_Label_2D_3Class_RGB_3Slice.csv",
                          "Parotids11Slice_Uncropped":stringTrainingDirectory+"\Training_Label_2D_Parotids_2Channel_11Slice.csv",
                          "RightParotids11Slice_Aug":stringTrainingDirectory[:31]+"\Training_Label_2D_RightParotids_2Channel_11Slice_Aug.csv",
                          "2Class2Channel3D_Uncropped":stringTrainingDirectory+"\Training_Label_3D_2Class_2Channel.csv",
                          "3Class2Channel3D_Uncropped":stringTraining3DDirectory+"Labels/",
                          "3Class2Channel3D":stringTraining3DDirectory[:33]+"Features/",
                          "3Class3Channel3Slice":stringTrainingDirectory+"\Training_Label_2D_3Class_3Channel_3Slice.csv",
                          "5Class2Channel3Slice":stringTrainingDirectory+"\Training_Label_2D_5Class_2Channel_3Slice.csv",
                          "5Class3Channel3Slice":stringTrainingDirectory+"\Training_Label_2D_5Class_3Channel_3Slice.csv",
                          }
    
    return TrainingFeaturesDict,TrainingLabelsDict

def get_Model_Dictionary():
    stringImageDirectory ="C:/Users/Oliver/Documents/University/Year 4/MPhys Project/Images/"
    stringModelDirectory = "D:/Models/Uncropped"
    ModelDict = {"3Class2Channel3Slice":stringModelDirectory[:9]+"/2D_3Class_2Channel_3Slice_Model",
                 "3Class2Channel3Slice_OneLayer":stringModelDirectory[:9]+"/2D_3Class_2Channel_3Slice_OneLayer_Model",
                 "3Class3Channel3Slice":stringModelDirectory[:9]+"/2D_3Class_3Channel_3Slice_Model",  
                 "3ClassRGB3Slice":stringModelDirectory[:9]+"/2D_3Class_RGB_3Slice_Model",
                 "3Class2Channel3Slice_Uncropped":stringModelDirectory+"/2D_3Class_2Channel_3Slice_Model", 
                 "3Class3Channel3Slice_Uncropped":stringModelDirectory+"/2D_3Class_3Channel_3Slice_Model",  
                 "3ClassRGB3Slice_Uncropped":stringModelDirectory+"/2D_3Class_RGB_3Slice_Model",
                 "Parotids11Slice_Uncropped":stringModelDirectory+"/2D_Parotids_2Channel_11Slice_Model",
                 "RightParotids11Slice_Aug":stringModelDirectory+"/2D_RightParotids_2Channel_11Slice_Aug_Model",
                 "2Class2Channel3D_Uncropped":stringModelDirectory+"/3D_2Class_2Channel_Model",
                 "3Class2Channel3D_Uncropped":stringModelDirectory+"/3D_3Class_2Channel_Model",
                 
                 "3Class3SliceSegmentation_Uncropped":stringModelDirectory+"/2D_3Class_3Slice_SegmentationModel",
                 "Parotids11SliceSegmentation_Uncropped":stringModelDirectory+"/2D_Parotids_11Slice_SegmentationModel"
                 }
    return stringImageDirectory, ModelDict