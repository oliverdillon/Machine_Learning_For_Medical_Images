import numpy as np
import psutil
import os
import keras
       
class Data_Generator_3D(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_index = idx * self.batch_size:(idx + 1) * self.batch_size
        
        batch_IDs_X = [self.list_IDs[k] for k in indx]
        
        batch_x, batch_y = Three_D_Data_Generator(batch_IDs,batch_IDs)

        return np.array(batch_x), np.array(batch_y)
   
    def on_epoch_end(self):
        'Shuffles after each epoch'
        np.random.shuffle(self.indexes) 
    
    
    def Get_Input(filename):
        img = np.load(filename)
        return img

    def Get_Output(filename):    
        labels = np.genfromtxt(filename, delimiter = ',')
        return labels
       
    def Three_D_Data_Generator(self,feature_files,label_files):
        #Pick out the files for the batch
        FeatureFilesForBatch = np.random.choice(feature_files, BatchSize, replace = False)
        #Pick out the files for the batch
        LabelFilesForBatch = np.random.choice(label_files, BatchSize, replace = False)
        BatchX = []
        BatchY = []
        
        for i,file in enumerate(feature_files):
            BatchX.append(self.Get_Input(file))
        for i,file in enumerate(label_files):
            BatchY.append(self.Get_Output(file))
        BatchX = np.array(BatchX)
        BatchY = np.array(BatchY)
        
        process = psutil.Process(os.getpid())
        print("Memory usage: {:.2%}".format(process.memory_percent()))
        print(BatchX.shape)
        
        #like return but for data generators
        return (np.array(BatchX),BatchY)
    
    
    
def Predict_Data_Generator(feature_files,index):
    while True:
        yield [np.load(feature_files[index])]
        
## User Guide:
#   when fitiing to data, use model.fit_generator()
#   files is a list of filepaths for all saved data
#   LabelDict should be a dictionary where the labels are the folder names and the data is in the form [0,0,0,0,0]
#   BatchSize is an integer