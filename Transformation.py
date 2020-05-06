# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:33:11 2020

@author: Oliver
"""
import random
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw
########################## SPLIT X/Y AND GET ORGINS ##############################
def get_coordinates(Points):
    x = [point[0] for point in Points]
    y = [point[1] for point in Points]

    ox = sum(x)/len(x)
    oy = sum(y)/len(y)
    
    return x, y, ox, oy
########################## ROTATION ##############################
def rotate_around_point(Points):
    x, y, ox, oy = get_coordinates(Points)
    ContourPoints = []
    
    radians = random.uniform(5,10)
    radians = radians*(-1)**random.randint(1,2)
    radians = radians*math.pi/180
    
    for i in range(len(x)):   
        qx = ox + math.cos(radians) * (x[i] - ox) + math.sin(radians) * (y[i] - oy)
        qy = oy + -math.sin(radians) * (x[i] - ox) + math.cos(radians) * (y[i] - oy)
        ContourPoints.append((qx,qy))
        
    return ContourPoints
########################## BULK TRANSLATION ##############################
def translate_points(Points):
    x, y, ox, oy = get_coordinates(Points)
    ContourPoints = []
    
    x_Translation= random.uniform(2,5)
    y_Translation= random.uniform(2,5)
    
    x_Translation= x_Translation*(-1)**random.randint(1,2)
    y_Translation= y_Translation*(-1)**random.randint(1,2)
    
    for i in range(len(x)):   
        qx = x[i] + x_Translation
        qy = y[i] + y_Translation
        ContourPoints.append((qx,qy))
    return ContourPoints
########################## TRANSLATION of CONTOURS on a Slice ##############################
def translate_contour(Points,x_Translation,y_Translation):
    x, y, ox, oy = get_coordinates(Points)
    ContourPoints = []
    
    for i in range(len(x)):   
        qx = x[i] + x_Translation
        qy = y[i] + y_Translation
        ContourPoints.append((qx,qy))
    return ContourPoints
########################## TRANSLATION of a  proportion of CONTOURS on a Slice ##############################
def MoveRandomPoints(Points,Percentage):
    x, y, ox, oy = get_coordinates(Points)
    ContourPoints =[]
    indices = []
    i = 0
    NumberOfPoints =len(x)
    
    while (i<NumberOfPoints*Percentage):
        randomIndex = random.randint(0,NumberOfPoints-1)
        randomcluster = random.randint(randomIndex+1,NumberOfPoints)
        for k in range(randomIndex,randomcluster):
            if not (k in indices):
                indices.append(k)
                i=i+1
    for j in indices:        
        x_Translation= random.uniform(2,5)*(-1)**random.randint(1,2)
        y_Translation= random.uniform(2,5)*(-1)**random.randint(1,2)
        
        x[j] = x[j] + x_Translation
        y[j] = y[j] + y_Translation
    
    for i in range(len(x)):  
        ContourPoints.append((x[i],y[i]))
        
    return ContourPoints
########################## RESIZE ##############################
def resize_points(Points):
    x, y, ox, oy = get_coordinates(Points)
    ContourPoints = []
    
    ScalingFactor = random.randrange(5,20,1)/10
    
    dx = [(point-ox) for point in x]
    dy = [(point-oy) for point in y]
    
    r=[]
    theta=[]
    
    for i in range(len(x)):
        r.append(np.sqrt(math.pow(dx[i],2)+math.pow(dy[i],2)))
        theta.append(np.arctan2(dy[i], dx[i]))
        
    ContourPoints = []
    
    for i in range(len(x)):
        qx = ox + (ScalingFactor * r[i] * np.sin(theta[i]))
        qy = oy + (ScalingFactor * r[i] * np.cos(theta[i]))
        ContourPoints.append((qx,qy))
    
    return ContourPoints

########################## SHEAR ##############################
def shear_points(Points):
    x, y, ox, oy = get_coordinates(Points)
    ContourPoints = []
    
    dx = [(point-ox) for point in x]
    dy = [(point-oy) for point in y]
    
    phi = random.uniform(50,70) * math.pi/180
    M = 1.0 / np.tan(phi)
    
    XorY = random.randint(0,1)    
    ContourPoints = []
    
    if XorY == 0:
        for i in range(len(x)):
            qx = x[i] + (M * dy[i])
            qy = y[i]
            ContourPoints.append((qx,qy))            
    
    else:
        for i in range(len(y)):
            qx = x[i]
            qy = y[i] + (M * dx[i])
            ContourPoints.append((qx,qy))
    
    return ContourPoints

########################## OBTAIN CONTOURED CT IMAGES ##############################
def FillContourArea(Vertices):
        '''
        FilledImage = np.zeros((512,512))
        Vertices = np.asarray(Vertices).reshape(-1,1,2)
        cv2.fillPoly(FilledImage, [Vertices], color = (255,0,255))
        return np.uint8(FilledImage)
        
        skimage.draw.polygon2mask(image_shape,polygon)
        '''
        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(512, 512), color=0)
        ImageDraw.Draw(img).polygon(xy=Vertices, outline=0, fill=1)
        #img = img.transpose(Image.ROTATE_90)
        mask = np.array(img).astype(bool)
    
        return np.uint8(mask)*255