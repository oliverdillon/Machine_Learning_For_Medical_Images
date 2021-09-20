import numpy as np
import psutil
import os

def Get_Input(filename):
    img = np.load(filename)
    return img

def Get_Output(filename):    
    labels = np.genfromtxt(filename, delimiter = ',')
    return labels

def Three_D_Data_Generator(feature_files,label_files, BatchSize):
    while True:
        #Pick out the files for the batch
        FeatureFilesForBatch = np.random.choice(feature_files, BatchSize, replace = False)
        #Pick out the files for the batch
        LabelFilesForBatch = np.random.choice(label_files, BatchSize, replace = False)
        BatchX = []
        BatchY = []
        
        for file in FeatureFilesForBatch:
            BatchX.append(Get_Input(file))
        for file in LabelFilesForBatch:
            BatchY.append(Get_Output(file))
        BatchX = np.array(BatchX)
        BatchY = np.array(BatchY)
        
        process = psutil.Process(os.getpid())
        print("Memory usage: {:.2%}".format(process.memory_percent()))
        print(BatchX.shape)
        
        #like return but for data generators
        yield (np.array(BatchX),BatchY)

def Predict_Data_Generator(feature_files,index):
    while True:
        yield [Get_Input(feature_files[index])]
        
## User Guide:
#   when fitiing to data, use model.fit_generator()
#   files is a list of filepaths for all saved data
#   LabelDict should be a dictionary where the labels are the folder names and the data is in the form [0,0,0,0,0]
#   BatchSize is an integer