import keras
from numpy.random import randn
from matplotlib import pyplot as plt
import  tensorflow as tf
from utility_functions import build_directory

#create GAN class
class GAN(keras.Model):
    #specify settings
    def __init__(self, d_model, g_model, results_directory, noise_size = 62, relu_alpha = 0.1):
        super(GAN, self).__init__()
        #d_model
        self.d_model = d_model
        #g_model
        self.g_model = g_model
        #alpha for activation functions
        self.relu_alpha = relu_alpha
        #noise size
        self.noise_size = noise_size
        #directory where generated images will be stored
        self.results_directory = results_directory

    #GAN compiler
    def compile(self, d_optimizer, g_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    #generate input
    def create_gen_input(self, batch_size, noise_size, seed=None):
        # create noise input and return
        noise = tf.random.normal([batch_size, noise_size], seed=seed)
        return  noise

    #training step
    def train_step(self, real_image_batch):
        #build necessary directories
        build_directory(self.results_directory)
        build_directory(f'{self.results_directory}/output')
        build_directory(f'{self.results_directory}/models_scores')
        # Define loss function
        binary_loss = keras.losses.BinaryCrossentropy()
        # Half-batch for training discriminator and batch for training generator
        batch = tf.shape(real_image_batch)[0]
        # Create generator input
        g_input = self.create_gen_input(batch, self.noise_size, seed=None)
        #training step for discriminator
        with tf.GradientTape() as d_tape:
            self.d_model.trainable = True
            d_tape.watch(self.d_model.trainable_variables)
            # Train discriminator using half batch real images
            y_disc_real = tf.ones((batch, 1))
            d_real_output = self.d_model(real_image_batch, training=True)
            d_loss_real = binary_loss(y_disc_real, d_real_output)
            # Train discriminator using half batch fake images
            y_disc_fake = tf.zeros((batch, 1))
            # Create fake image batch
            fake_image_batch = self.g_model(g_input, training=True)
            d_fake_output = self.d_model(fake_image_batch, training=True)
            d_loss_fake = binary_loss(y_disc_fake, d_fake_output)
            d_loss = d_loss_real + d_loss_fake
        # Calculate gradients
        d_gradients = d_tape.gradient(d_loss, self.d_model.trainable_variables)
        # Optimize
        self.d_optimizer.apply_gradients(zip(d_gradients, self.d_model.trainable_variables))
        with tf.GradientTape() as g_tape:
            # Create generator input
            g_noise = self.create_gen_input(batch*2, self.noise_size, seed=None)
            g_input = g_noise
            g_tape.watch(self.g_model.trainable_variables)
            # Create fake image batch
            fake_image_batch = self.g_model(g_input, training=True)
            d_fake_output = self.d_model(fake_image_batch, training=True)
            # Generator Image loss
            y_gen_fake = tf.ones((batch*2, 1))
            g_img_loss = binary_loss(y_gen_fake, d_fake_output)
            # Generator total loss
            g_loss = g_img_loss

        # Calculate gradients
        # do not want to modify the neurons in the discriminator when training the generator and the auxiliary model
        self.d_model.trainable=False
        g_gradients = g_tape.gradient(g_loss, self.g_model.trainable_variables)

        # Optimize
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_model.trainable_variables))


        return {"d_loss_real": d_loss_real, "d_loss_fake": d_loss_fake, "g_loss": g_loss }
