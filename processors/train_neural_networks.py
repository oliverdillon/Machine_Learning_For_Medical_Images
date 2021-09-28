from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D
from tensorflow.keras.layers import MaxPooling3D,Conv3D,Dropout
from tensorflow.keras.optimizers import Adam
from models.training_image_loader import Training_image_loader
class Train_neural_network:
    def __init__(self,training_data,testing_data,index1,index2):
        self.allowed_organs = ["Right Parotid","Left Parotid"]
        self.no_classes = len(self.allowed_organs)
        self.training_data = training_data
        self.testing_data = testing_data
        self.stepSize = 2
        self.epoch_no = 15

        self.training_steps_per_epoch = int(round(len(self.X_train))/self.stepSize)
        self.validation_steps_per_epoch = int(round(len(self.X_Val))/self.stepSize)

        self.X_train =self.training_data.dataset[index2:]
        self.y_train =self.training_data.labels[index2:]
        self.X_Val =self.training_data.dataset[index1:index2]
        self.y_Val =self.training_data.labels[index1:index2]


    def build_3D_model(no_classes,shape,layers = ["Conv3D","Maxpool3D","Conv3D","Maxpool3D","Dense"],FilterNumbers= [32,0,64,0,128],lr=0.001):
        if(len(layers)!=len(FilterNumbers)):
            print("Number of layers must match the number of filters")
            return
        else:
            # Initialising the CNN
            model = Sequential() #add each layer in turn

            #Choose layers
            for i,layer in enumerate(layers):
                if(layer == "Conv3D1"):
                    model.add(Conv3D(FilterNumbers[i], (1, 1, 1), input_shape = shape, activation = 'relu'))
                if(layer == "Conv3D2"):
                    model.add(Conv3D(FilterNumbers[i], (2, 2, 2), input_shape = shape, activation = 'relu'))
                if(layer == "Conv3D"):
                    model.add(Conv3D(FilterNumbers[i], (3, 3, 3), input_shape = shape, activation = 'relu'))
                if(layer == "Conv3D4"):
                    model.add(Conv3D(FilterNumbers[i], (4, 4, 3), input_shape = shape, activation = 'relu'))
                if(layer == "Conv3D5"):
                    model.add(Conv3D(FilterNumbers[i], (5, 5, 3), input_shape = shape, activation = 'relu'))
                if(layer == "Conv3D11"):
                    model.add(Conv3D(FilterNumbers[i], (5, 5, 3), input_shape = shape, activation = 'relu'))
                if(layer == "Maxpool3D"):
                    model.add(MaxPooling3D(pool_size = (2, 2, 2)))
                if(layer == "Maxpool3D3"):
                    model.add(MaxPooling3D(pool_size = (3, 3, 2)))
                if(layer == "Dropout"):
                    #Dropout: 0.25 after maxpool? 0.5 after Dense?
                    model.add(Dropout(FilterNumbers[i]))
                if(layer == "Dense"):
                    # Flattening for Dense Layer
                    model.add(Flatten())
                    # Full Connected "Dense" Layer
                    model.add(Dense(units = FilterNumbers[i], activation = 'relu'))

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

    def train_neural_network(self):
        training_generator = Training_image_loader(self.X_train, self.y_train, self.stepSize)
        validation_generator = Training_image_loader(self.X_Val, self.y_Val, self.stepSize)

        FilterNumbers = [16,0,16,16,16,0,16,16,512]
        layers = ["Conv3D","Maxpool3D","Conv3D1","Conv3D1","Conv3D","Maxpool3D","Conv3D1","Conv3D1","Dense"]
        model = self.build_3D_model(self.no_classes,self.training_data.dataset[0].shape,layers,FilterNumbers)
        history = model.fit_generator(generator=training_generator, steps_per_epoch =self.training_steps_per_epoch,
                              validation_data=validation_generator, validation_steps=self.validation_steps_per_epoch,
                              epochs = self.epoch_no, verbose = 2, max_queue_size =1)