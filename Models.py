# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:04:58 2020

@author: Oliver
"""
# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D
from tensorflow.keras.layers import MaxPooling3D,Conv3D,Dropout
from tensorflow.keras.layers import Input,concatenate,UpSampling2D,Conv2DTranspose
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

################################# BUILDING 2D CNN #################################
def build_2D_model(no_classes,shape,lr=0.001,FilterNumbers= [32,64,128]):
    # Initialising the CNN
    model = Sequential() #add each layer in turn

    # First Convolutional Layer
    model.add(Conv2D(32, (3, 3), input_shape = shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Second Convolutional layer
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))  

    # Flattening for Dense Layer
    model.add(Flatten())
    
    # Full Connected "Dense" Layer
    model.add(Dense(units = 128, activation = 'relu'))
    
    
    if (no_classes == 2):
        activ = 'sigmoid'
        loss_func = 'binary_crossentropy'
    else:
        activ = 'softmax'
        loss_func = 'categorical_crossentropy'
    
    #Output Layer
    model.add(Dense(units = no_classes, activation = activ))

    optimize = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # Compiling the CNN
    model.compile(optimizer = optimize, loss = loss_func, metrics = ['accuracy'])
    
    print(model.summary())
    return model

def build_3D_model(no_classes,shape,lr=0.001,FilterNumbers= [32,64,128],Drop = True):
    # Initialising the CNN
    model = Sequential() #add each layer in turn

    # First Convolutional Layer
    model.add(Conv3D(FilterNumbers[0], (3, 3, 3), input_shape = shape, activation = 'relu'))
    model.add(MaxPooling3D(pool_size = (2, 2, 2)))
    
    if(Drop):
        model.add(Dropout(0.25))

    # Second Convolutional layer
    model.add(Conv3D(FilterNumbers[1], (3, 3, 3), activation = 'relu'))
    model.add(MaxPooling3D(pool_size = (2, 2, 2)))
    
    if(Drop):
        model.add(Dropout(0.25))

    # Flattening for Dense Layer
    model.add(Flatten())

    # Full Connected "Dense" Layer
    model.add(Dense(units = FilterNumbers[2], activation = 'relu'))
    
    if(Drop):
        model.add(Dropout(0.8))

    if (no_classes == 2):
        activ = 'sigmoid'
        loss_func = 'binary_crossentropy'
    else:
        activ = 'softmax'
        loss_func = 'categorical_crossentropy'
    
    #Output Layer
    model.add(Dense(units = no_classes, activation = activ))
    
    optimize = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # Compiling the CNN
    model.compile(optimizer = optimize, loss = loss_func, metrics = ['accuracy'])
    
    print(model.summary())
    return model

def build_segmentation_model(shape):
    no_classes = 2
    #concat = Concatenate()

    # Initialising the CNN
    inputs =Input(shape)
    #inputs =Input((512,512,1))

    ######## ENCODING ########
    # First Convolutional Layer
    Conv1 = Conv2D(32, (3, 3), activation = 'relu',padding = 'same')(inputs)
    Conv2 = Conv2D(32, (3, 3), activation = 'relu',padding = 'same')(Conv1)
    Maxpooling1 = MaxPooling2D(pool_size = (2,2))(Conv2)
    Maxpooling1 = Dropout(0.25)(Maxpooling1)

    # Second Convolutional layer
    Conv3 = Conv2D(64, (3, 3), activation = 'relu',padding = 'same')(Maxpooling1)
    Conv4 = Conv2D(64, (3, 3), activation = 'relu',padding = 'same')(Conv3)
    Maxpooling2 = MaxPooling2D(pool_size = (2,2))(Conv4)
    Maxpooling2 = Dropout(0.5)(Maxpooling2)

    # Third Convolutional layer
    Conv5 = Conv2D(128, (3, 3), activation = 'relu',padding = 'same')(Maxpooling2)
    Conv6 = Conv2D(128, (3, 3), activation = 'relu',padding = 'same')(Conv5)
    Maxpooling3 = MaxPooling2D(pool_size = (2,2))(Conv6)
    Maxpooling3 = Dropout(0.5)(Maxpooling3)

    # Fourth Convolutional layer
    Conv7 = Conv2D(256, (3, 3), activation = 'relu',padding = 'same')(Maxpooling3)
    Conv8 = Conv2D(256, (3, 3), activation = 'relu',padding = 'same')(Conv7)
    Maxpooling4 = MaxPooling2D(pool_size = (2,2))(Conv8)

    ######## MIDDLE ########
    # Convolutional layer
    ConvMIDDLE1 = Conv2D(512, (3, 3), activation = 'relu',padding = 'same')(Maxpooling4)
    ConvMIDDLE2 = Conv2D(512, (3, 3), activation = 'relu',padding = 'same')(ConvMIDDLE1)
    UpSampling1 = Conv2DTranspose(512, (3, 3), strides=(2, 2), activation = 'relu',padding = 'same')(ConvMIDDLE2)

    ######## DECODING ########
    # First Deconvolutional Layer
    Skip1 = concatenate([UpSampling1,Conv8])#First Skip Node
    Skip1 = Dropout(0.5)(Skip1)
    Deconv1 = Conv2D(256, (3, 3), activation = 'relu',padding = 'same')(Skip1)
    Deconv2 = Conv2D(256, (3, 3), activation = 'relu',padding = 'same')(Deconv1)
    UpSampling2 = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation = 'relu',padding = 'same')(Deconv2)

    # Second Deconvolutional Layer
    Skip2 = concatenate([UpSampling2,Conv6])#Second Skip Node
    Skip2 = Dropout(0.5)(Skip2)
    Deconv3 = Conv2DTranspose(128, (3, 3), activation = 'relu',padding = 'same')(Skip2)
    Deconv4 = Conv2DTranspose(128, (3, 3), activation = 'relu',padding = 'same')(Deconv3)
    UpSampling3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation = 'relu',padding = 'same')(Deconv4)

    # Third Deconvolutional Layer
    Skip3 = concatenate([UpSampling3,Conv4])#Third Skip Node
    Skip3 = Dropout(0.5)(Skip3)
    Deconv5 = Conv2DTranspose(64, (3, 3), activation = 'relu',padding = 'same')(Skip3)
    Deconv6 = Conv2DTranspose(64, (3, 3), activation = 'relu',padding = 'same')(Deconv5)
    UpSampling4 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation = 'relu',padding = 'same')(Deconv6)

    # Fourth Deconvolutional Layer
    Skip4 = concatenate([UpSampling4,Conv2])#Fourth Skip Node
    Skip4 = Dropout(0.5)(Skip4)
    Deconv10 = Conv2D(32, (3, 3), activation = 'relu',padding = 'same')(Skip4)
    Deconv11 = Conv2D(32, (3, 3), activation = 'relu',padding = 'same')(Deconv10)

    ######## FINALISING ########
    ConvFinal = Conv2D(no_classes+1, (1, 1), activation = 'softmax',padding = 'same')(Deconv11)
    model = Model(inputs=inputs,outputs =ConvFinal)

    # Compiling the CNN
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])#'binary_crossentropy'
    print(model.summary())
    
    return model

def n_fold_Validation(X,y,Epoch_No,n,Model_Type):
    #n-fold cross validation
    num_val_samples =len(X)/n
    all_acc=[]
    all_loss=[]
    all_mae_history =[]
    
    for i in range (n):
        print ("Fold:%2i"%i)
        
        Start_Partition_Index= i*num_val_samples
        End_Partition_Index= (i+1)*num_val_samples
        val_X = X[Start_Partition_Index:End_Partition_Index]
        val_y = X[Start_Partition_Index:End_Partition_Index]
        
        
        train_X = np.concatenate([ X[:Start_Partition_Index], X[End_Partition_Index:]],axis=0)
        train_y = np.concatenate([ y[:Start_Partition_Index], y[End_Partition_Index:]],axis=0)
        
        if(Model_Type == "2D"):
            model = build_2D_model()
        if(Model_Type == "2D"):
            model = build_3D_model()
        
        history = model.fit(train_X,train_y,validation_data = (val_X,val_y),batch_size = 16, epochs = Epoch_No)
        
        loss_history = history.history['val_loss']
        accuracy_history = history.history['val_acc']
        
        all_mae_history.append(mae_history)
        
        val_loss, val_acc = model.evaluate(val_X,val_y,verbose=0)
        all_acc.append(val_acc)
        all_loss.append(val_loss)
        
################################# PLOT MODEL ACTIVATIONS #################################
def plot_Model_Activations(new_model,key_Dict_Model,key_Dict,XTest):
    # Extracts the outputs of the top 4 layers
    layer_outputs = [layer.output for layer in new_model.layers[:4]] 
    layer_names = [layer.name for layer in new_model.layers[:4]]
    
    #Creates a model that will return these outputs, given the model input
    activation_model = models.Model(inputs=new_model.input, outputs=layer_outputs)
    activation_model.trainable = False
    
    
    noOfTests = 1
    scale = noOfTests*5
    if(key_Dict.find("RGB")!=-1):  
        fig1,ax1 = plt.subplots(noOfTests,3,figsize=[10,scale])
    else:
        fig1,ax1 = plt.subplots(noOfTests,3,figsize=[13,scale])
    
    #Loop adds to plots
    k = 0
    
    for index in range (0,noOfTests):
        #image predictions
        test_image = np.expand_dims(XTest[index], axis = 0)
        
        #Returns a list of five Numpy arrays: one array per layer activation
        activations = activation_model.predict(test_image) 
        
        #get activation matrices of layers
        first_layer_activation = activations[1]
        first_layer_n_features = first_layer_activation.shape[-1]
        #second_layer_activation = activations[3]
        #second_layer_n_features = second_layer_activation.shape[-1]
    
        #Sum contributions of activations after first Layer
        first_conv_total = np.zeros(first_layer_activation[0, :, :, 0].shape)
        for i in range (0,first_layer_n_features):
            first_conv_total+=first_layer_activation[0, :, :, i]
        """
        #Sum contributions of activations after second Layer
        second_conv_total = np.zeros(second_layer_activation[0, :, :, 0].shape)
        for i in range (0,second_layer_n_features):
            second_conv_total+=second_layer_activation[0, :, :, i]
          """  
        #Plot graph
        if(key_Dict.find("RGB")!=-1):
            ax1[k,0].imshow(XTest[index])
            ax1[k,0].axis('off')
            ax1[k,1].imshow(first_conv_total, cmap='gray')
            ax1[k,1].axis('off') 
            ax1[k,2].imshow(second_conv_total, cmap='gray')
            ax1[k,2].axis('off') 
        else:
            ax1[k,0].imshow(XTest[index][..., 0], cmap='gray')
            ax1[k,0].axis('off')
            ax1[k,1].imshow(first_conv_total, cmap='gray')
            ax1[k,1].axis('off')
            #ax1[k,2].imshow(second_conv_total, cmap='gray')
            #ax1[k,2].axis('off') 
            ax1[k,2].imshow(XTest[index][..., 1], cmap='gray')
            ax1[k,2].axis('off')
        
        k+=1  
    
    ax1[0,0].set_title("CT Image")
    ax1[0,1].set_title("First Layer")
    #ax1[0,2].set_title("Second Layer")
    if(key_Dict.find("RGB")==-1):
        ax1[0,2].set_title("Contour")
    
    stringImageDirectory ="C:/Users/Oliver/Documents/University/Year 4/MPhys Project/Images/Semester 2/"
    fig1.savefig(stringImageDirectory+key_Dict_Model+"_Activation_Ten.png")
    plt.show()