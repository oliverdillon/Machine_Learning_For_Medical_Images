#
# Spatial pyramidal pooling layer for 3D volumes using the Tensorflow backend

# author: Alex Fish

# heavily based on keras-spp by yhenon
# https://github.com/yhenon/keras-spp/blob/master/spp/SpatialPyramidPooling.py
#

from tensorflow.keras import layers
import keras.backend as keras

class SPP3D(layers.Layer):
    #Spatial pyramidal pooling layer for 3D voxel - based inputs
    
    # inputs:
    #   pools  - list of pooling regions to use
    #Input Shape - 5D tensor using the TF order
    #Output Shape - 2D tensor for passing to the dense layers
    
    def __init__ (self, pools):
        super(SPP3D, self).__init__()
        self.pools = pools
        self.OutputsPerChannel = [i**2 for i in pools]
        
        
    
    def build(self, input_shape):
        self.nb_channels = input_shape[4]
        print(input_shape)
        super(SPP3D, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.nb_channels * self.OutputsPerChannel)
        
    def get_config(self):
        config = {'pools':self.pools}
        base_config = super(SPP3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, x, mask = None):
        
        input_shape = keras.shape(x)
        
        num_rows = input_shape[1]
        num_cols = input_shape[2]
        num_layers = input_shape[3]
        
        rows_length = [keras.cast(num_rows, 'float32') / i for i in self.pools]
        cols_length = [keras.cast(num_cols, 'float32') / i for i in self.pools]
        layers_length = [keras.cast(num_layers, 'float32') / i for i in self.pools]
        
        outputs = []
        
        for pool_num, num_pool_regions in enumerate(self.pools):
            for kz in range(num_pool_regions):
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = ix*cols_length[pool_num]
                        x2 = ix*cols_length[pool_num] + cols_length[pool_num]
                        
                        y1 = jy*rows_length[pool_num]
                        y2 = jy*rows_length[pool_num] + cols_length[pool_num]
                        
                        z1 = kz*layers_length[pool_num]
                        z2 = kz*layers_length[pool_num] + cols_length[pool_num]
                        
                        new_shape = [input_shape[0], y2-y1, x2-x1 , z2-z1, input_shape[4]]
                        
                        x_crop = x[0:-1, y1:y2, x1:x2, z1:z2, 0:-1]
                        xm = keras.reshape(x_crop, new_shape)
                        pooled_val = keras.max(xm, axis=(1,2,3))
                        outputs.append(pooled_val)
        
        outputs = keras.concatenate(outputs)
        
        return outputs                        
