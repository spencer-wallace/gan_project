import tensorflow as tf
import keras
from PIL import Image, ImageStat
import numpy as np
from utility_functions import build_directory
from numpy.random import randn


#callbacks class that generates an image after each epoch to follow development
class ImageTracer(tf.keras.callbacks.Callback):
    #necessary info for the class
    def __init__(self, results_directory, noise_size = 62):
        #directory where image should be saved
        self.results_directory = results_directory
        #size of noise input
        self.noise_size = noise_size
    def on_epoch_end(self, epoch, logs=None):
        #make sure that directory exists
        build_directory( f'{self.results_directory}/output')
        #seems to be having weird effect on images, so have switched back to random noise
        #img = np.squeeze(self.g_model_continuous.predict(self.input).astype(np.uint8))
        #random noise is fed into g_model for a prediction which is then converted to  a PIL  image and saved
        img = np.squeeze(self.model.g_model.predict(randn(self.noise_size).reshape(1,self.noise_size)*255).astype(np.uint8))
        Img = Image.fromarray(img)
        out_path = f'{self.results_directory}/output/train_out_epoch_{epoch+1}.jpg'
        Img.save(out_path)


#callback which traces the standard deviation of the color of a grayscale version of prediction image
#sometimes the gan collapses into outputing single color, this allows the function to brake in that case
class ColorSigmaBreaker(tf.keras.callbacks.Callback):
    def __init__(self, noise_size = 62, sigma = 7 , patience = 5):
        #noise size of input
        self.noise_size = noise_size
        #number of epochs required to trigger stop
        self.patience = patience
        #number of standard deviations of color that will trigger a wait count
        self.sigma = sigma
        #count of how many epochs in a row the standard deviation was below the sigma
        self.wait = 0
    def on_epoch_end(self, epoch, logs=None):
        #make prediction
        img = np.squeeze(self.model.g_model.predict(randn(self.noise_size).reshape(1,self.noise_size)*255).astype(np.uint8))
        #convert prediciton to grayscale
        Img = Image.fromarray(img).convert('L')
        #get standard deviation of the predicted image
        current_sigma = ImageStat.Stat(Img)._getstddev()
        #print sigmas to know whats happening
        print(current_sigma, self.sigma)
        #if sigma is less than current sigma, reset wait to 0
        if np.less(self.sigma, current_sigma):
            self.wait = 0
        #if current sigma is less than set sigma, add 1 to wait counter
        #if wait count goes above set patience, stop training
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

#create intial image to demonstrate how simplified version of model processes the noise into an image as a baseline image
class InitialImage:
    def __init__(self, g_model, noise_size, x_input, y_input, rgb, results_directory, kernel_size):
        #settings taken directly from the settings for the gan's d and g models
        self.g_model = g_model
        self.noise_size = noise_size
        self.x_input = x_input
        self.y_input = y_input
        self.kernel_size = kernel_size
        self.rgb = rgb
        self.results_directory = results_directory
    #build simplified version of the model
    def build(self):
        self.input_model = keras.models.Sequential()
        self.input_model.add(keras.layers.Input(shape=(self.noise_size, )))
        self.input_model.add(keras.layers.Dense(units=self.x_input*self.y_input*64, use_bias=False))
        self.input_model.add(keras.layers.Reshape(target_shape=( self.y_input,self.x_input, 64)))
        if self.rgb ==1:
            self.input_model.add(keras.layers.Conv2DTranspose(3, kernel_size=(self.kernel_size, self.kernel_size),
                                                        strides=(1, 1), padding="same", activation="tanh"))
        elif self.rgb ==0:
            self.input_model.add(keras.layers.Conv2DTranspose(1, kernel_size=(self.kernel_size, self.kernel_size),
                                                    strides=(1, 1), padding="same", activation="tanh"))

    #use simplified model to generate first image and save
    def generate(self):
        img = np.squeeze(self.input_model.predict(randn(self.noise_size).reshape(1,self.noise_size)*255).astype(np.uint8))
        Img = Image.fromarray(img)
        out_path = f'{self.results_directory}/output/train_out_epoch_0.jpg'
        Img.save(out_path)
