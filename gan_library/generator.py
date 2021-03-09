import tensorflow as tf
import keras
import numpy as np

#create generator net as an object
class Generator:
    #establish settings for Generator
    def __init__(self, y_input, x_input,  n_convols =2,  n_filters=34, noise_size=62, rgb = 1,
                kernel_size = 4, strides = 2, relu_alpha = 0.1, dense_layers = None, dense_size=None):
        #vertical length of reshape from noise
        self.y_input = y_input
        #horizontal length of reshape from noise
        self.x_input = x_input
        #number of transpose convolutional blocks
        self.n_convols =n_convols
        #initial number of filters in transpose convolutional block, will be halved for each block
        self.n_filters=n_filters
        #size of noise inputed to generate images
        self.noise_size=noise_size
        #if images will be color or not. 0 = grayscale, 1 = color
        self.rgb = rgb
        #size of kernel for transpose convolutional layer
        self.kernel_size = kernel_size
        #stride of transpose convolutional layer
        self.strides = strides
        #activation alpha
        self.relu_alpha = relu_alpha
        #number of dense layers applied before transpose covolution
        self.dense_layers = dense_layers
        #number of neurons per dense layer
        self.dense_size = dense_size
    #build generator
    def build(self):
        #init sequential model
        self.model = keras.models.Sequential()
        #input layer of size of noise
        self.model.add(keras.layers.Input(shape=(self.noise_size, )))
        #if there are dense layers specified, adds them here, otherwise passes on to next step
        if self.dense_layers == None:
            pass
        else:
            for layer in range(self.dense_layers):
                self.model.add(keras.layers.Dense(units=dense_size, use_bias=False))
                self.model.add(keras.layers.BatchNormalization())
                self.model.add(keras.layers.ReLU(alpha = self.relu_alpha))
        # dense layer to create appropriate size for reshape
        self.model.add(keras.layers.Dense(units=self.x_input*self.y_input*64, use_bias=False))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ReLU())

        # reshape
        self.model.add(keras.layers.Reshape(target_shape=( self.y_input,self.x_input, 64)))
        #number of filters for first layer established
        self.nf = self.n_filters
        # first  layer
        self.model.add(keras.layers.Conv2DTranspose(self.nf, kernel_size=(self.kernel_size, self.kernel_size),
                                                    strides=(self.strides, self.strides), padding='same', use_bias=False))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.ReLU())
        # filters halved
        self.nf = self.nf//2
        #for any transposed convolutional layers specified beyond one, adds via a for loop, halving filters each time
        for u in range(2, self.n_convols+1):
            self.model.add(keras.layers.Conv2DTranspose(self.nf, kernel_size=(self.kernel_size, self.kernel_size),
                                                        strides=(self.strides, self.strides), padding='same', use_bias=False))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.ReLU())
            self.nf = self.nf//2
        #final layer, if color breaks into three color channels or remains single channel for grayscale. Uses  tanh activation as per best practives
        if self.rgb ==1:
            self.model.add(keras.layers.Conv2DTranspose(3, kernel_size=(self.kernel_size, self.kernel_size),
                                                        strides=(1, 1), padding="same", activation="tanh"))
        elif self.rgb ==0:
            self.model.add(keras.layers.Conv2DTranspose(1, kernel_size=(self.kernel_size, self.kernel_size),
                                                    strides=(1, 1), padding="same", activation="tanh"))

        return self.model
