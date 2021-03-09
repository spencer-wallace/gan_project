from descriminator import Discriminator
from generator import Generator
from data_get import get_data, process
from create_gan import GAN
from viz_tools import save_images
from callbacks import ImageTracer, ColorSigmaBreaker, InitialImage
from utility_functions import build_directory
import tensorflow as tf
import keras
from numpy.random import randn
import pickle
#class to instantiate gan as well as run the training with additional parameters
class GanFunc():
    #parameters for gan
    def __init__(self, results_directory, image_directory, x_input =20, y_input=20,
                n_convols=2, epochs=50, g_learn=0.0002,
                d_learn= 0.0002, gfilters = 128, relu_alpha = 0.2,
                batch_size = 10, rgb = 1, run_checkpoint = 1, noise_size = 62, kernel_size = 4, sigma = 7, patience = 4):
        self.image_directory= image_directory
        self.results_directory = results_directory
        self.x_input=x_input
        self.y_input=y_input
        self.n_convols = n_convols
        self.epochs = epochs
        self.g_learn = g_learn
        self.d_learn = d_learn
        self.gfilters = gfilters
        self.relu_alpha = relu_alpha
        self.batch_size =batch_size
        self.rgb = rgb
        self.noise_size = noise_size
        self.run_checkpoint = run_checkpoint
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.patience = patience

    #create gan
    def call(self):
        #create d and g models
        self.d_model= Discriminator(y_input= self.y_input, x_input = self.x_input, n_filters = self.gfilters/(2**self.n_convols),
                                    n_convols = self.n_convols, rgb=self.rgb).build()
        self.g_model= Generator(self.y_input, self.x_input, n_filters = self.gfilters, n_convols = self.n_convols, rgb =self.rgb).build()
        #create gan using d and g models
        self.infogan = GAN(self.d_model, self.g_model, self.results_directory,
                                        noise_size=62, relu_alpha = self.relu_alpha)
        #compile gan
        self.infogan.compile(d_optimizer=keras.optimizers.Adam(learning_rate=self.d_learn),
                    g_optimizer=keras.optimizers.Adam(learning_rate=self.g_learn))
        #get data ready
        self.X = get_data(self.image_directory, self.y_input,
                        self.x_input, self.batch_size,self.rgb, self.n_convols)
    #run train step as well as save and document the results and processes
    def run(self):
        #if details wanted, 1 is selected
        if self.run_checkpoint ==1:
            #necessary directories shoud have previously been built but double checked here
            build_directory(self.results_directory)
            build_directory(f'{self.results_directory}/output')
            #create initial image of what the noise looks like when run through a simplified version of the model for reference
            init_img = InitialImage(self.g_model, self.noise_size, self.x_input, self.y_input,
                                    self.rgb, self.results_directory, self.kernel_size)
            init_img.build()
            init_img.generate()
            #train model, custom callbacks used to generate an image after each epoch and break the training if image converges to blank unicolor image for too many epochs
            history = self.infogan.fit(self.X, epochs=self.epochs,
                                        callbacks =[ImageTracer(self.results_directory, self.noise_size ),
                                         ColorSigmaBreaker( noise_size = self.noise_size, sigma = self.sigma, patience = self.patience)] )
        #if 1 not selected, simply trains the model
        else:
            history = self.infogan.fit(self.X, epochs=self.epochs)

        #generate three images at the end
        save_images(self.g_model, self.results_directory, 3)
        #save model weights
        self.g_model.save(f'{self.results_directory}/models_scores/g_model.keras')
        self.d_model.save(f'{self.results_directory}/models_scores/d_model.keras')
        #save model scores
        with open(f'{self.results_directory}/models_scores/scores.dat', 'wb') as f:
            pickle.dump(history.history['d_loss_fake'],f)
            pickle.dump(history.history['d_loss_real'],f)
            pickle.dump(history.history['g_loss'], f)
