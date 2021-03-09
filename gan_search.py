from gan_library.gan_function import GanFunc
from gan_library.utility_functions import build_directory
import itertools
import os
#settings to run gan on
#number of epochs
epochs = [128]
#number of convolutional layers
convol = [1,2]
#d model learning rate
dlearn = [2e-4]
#g model learning rate
glearn = [2e-4, 2e-2]
#rgb or grayscale. rgb == 1, grayscale == 0
rgb = [1]
#initial number of filters for the g model. will also set the number for the d model by dividing that number by 2**number convolutional layers
gfilter = 256
#input image width
x_input = 256
#input image height
y_input = 256
#directory where images will be saved. By default will be a folder on the same level as this file called outputs
base_dir = 'outputs'
#directory where input data is stored. If using data provided and it has been unzipped inplace, data_directory will be 'vase_image_data'
data_directory = 'vase_image_data'
#batch size for training
batch_size = 5
#run the custom checkpoints
run_checkpoint = 1

#counters to help with if there are any errors
#index of list of products which starts at
start_num =0
#counter for how far the search has gone
current_num = 0

build_directory('outputs')

#create a list of possible combinations of settings, iterate over list from index of start_num
for i, xs in enumerate([i for i in itertools.product(epochs, convol, dlearn, glearn)][start_num:]):
    #instantiate ganfunc with settings from iteration, call and run
    gan = GanFunc(f'{base_dir}/batch_{current_num}', data_directory, epochs = xs[0],
                n_convols = xs[1],d_learn = xs[2], g_learn = xs[3], rgb = rgb, gfilters = gfilter, batch_size = batch_size,
                run_checkpoint =run_checkpoint, x_input= x_input, y_input = y_input)
    gan.call()
    print(xs)
    gan.run()
    #add one to current number counter
    current_num+=1
    #quick little readme generated to save settings
    read_me = f'epochs:{xs[0]}, n_convols:{xs[1]}, d_learn:{xs[2]}, g_learn:{xs[3]}'
    text = open(f'{base_dir}/batch_{i+start_num}/readme.txt', 'w')
    text.write(read_me)
    text.close()
