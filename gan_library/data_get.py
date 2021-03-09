import tensorflow as tf
import keras
import numpy as np
#function which standardizes image from 0 to 255 to 0 to 1
def process(image):
    image = tf.cast((image*2./255. -1.) ,tf.float32)
    return image
#data processing and fetching function
def get_data(data_dir, input_y, input_x, batch_size,rgb,n_convols):
    #get and batch data
    X= tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                          seed=123,
                                                          label_mode = None,
                                                          image_size=(input_y*2**n_convols, input_x*2**n_convols),
                                                          batch_size=batch_size )
    #convert all images to rgb if rgb desired or to grayscale for grayscale
    if rgb==1:
        X = X.map(lambda x: tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x)
    elif rgb ==0:
        X = X.map(lambda x: tf.image.rgb_to_grayscale(x) if x.shape[-1] == 3 else x)
    #standardize data using process function
    X= X.map(process)
    return X
