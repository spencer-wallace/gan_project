from PIL import Image, ImageOps
from numpy.random import randn
import os
import numpy as np
import imageio as io
from utility_functions import build_directory

#function to generate and show a given number of images
def show_images(model, n_images):
    for i in range(n_images):
        img = (np.squeeze(model.predict(randn(62)))*255).astype(np.uint8)
        Img = Image.fromarray(img)
        display(Img)
#function to generate and save a given number of images
def save_images(model,image_directory, n_images):
    try:
        os.mkdir( f'{image_directory}/output')
    except:
        pass
    for i in range(n_images):
        img = (np.squeeze(model.predict(randn(62).reshape(1,62)))*255).astype(np.uint8)
        Img = Image.fromarray(img)
        out_path = f'{image_directory}/output/output_{i}.jpg'
        Img.save(out_path)
#takes a directory and generates a gif from all the images with train in the title
def generate_gif(directory):
    outputs = [i for i in os.listdir(directory) if i.beginswith('train') and i.endswith('.jpg')]
    outputs = outputs.sort(key=lambda f: int(re.sub('\D', '', f)))
    imageio.mimsave('../animation/gif/movie.gif', images, duration = 1.5)

#add margins to an image, used to make annotation easier
def add_margin(path):
    img = Image.open(path)
    # border color
    color = "black"
    # top, right, bottom, left
    border = (20, 50,20, 50)
    new_img = ImageOps.expand(img, border=border, fill=color)
    return new_img
#annotate image with the settings of the gan used to generate it as well as its scores
def annotate_image(image,scores, characteristics):
#     image = Image.open('Focal.png')
    width, height = image.size

    draw = ImageDraw.Draw(image)
    textwidth, textheight = draw.textsize(scores)

    margin = 20
    x = 20
    y =  textheight+ margin
    draw.text((x, y), characteristics)

    x = 20
    y =  height - textheight- margin
    draw.text((x, y), scores)
    return image
#margin and annotate image
def margin_and_annotate(path, scores, characteristics, new_dir):
    image = add_margin(path)
    image = annotate_image(image, scores, characteristics)
    image.save(new_dir + path.split('/')[-1])
#margin, annotate and create gif
def annotated_gif(directory, n_convols):
    outs_dir = directory + '/output'
    new_dir = directory+'/annotated_output'
    build_directory(new_dir)
    inputs = [i for i in os.listdir(outs_dir) if i.startswith('train') and i.endswith('.jpg')]
    inputs.sort(key=lambda f: int(re.sub('\D', '', f)))
    inputs = [outs_dir +'/'+ j for j in inputs]
    with open(directory+'/readme.txt', 'r') as f:
        desired_info = [s.split(':')[1] for s in f.read().split(',')]
    desired_info = ' '.join(map(str, desired_info))
    with open(f'{directory}/models_scores/scores.dat', 'rb') as f:
        d_loss_fake = pickle.load(f)
        d_loss_real = pickle.load(f)
        g_loss = pickle.load(f)
    first = [0]
    d_loss_fake = first + d_loss_fake
    d_loss_real = first + d_loss_real
    g_loss = first + g_loss
    for u, o in enumerate(inputs[1:]):
        info_string = o.split('_')[-1].replace('.jpg', '')+ ' '
        info_string += desired_info
        score_string = f'dlf:{round(d_loss_fake[u], 2)} dlr:{round(d_loss_real[u],2)} gl:{round(g_loss[u], 2)}'
        margin_and_annotate(o, score_string, info_string, new_dir+'/')
    outputs = [i for i in os.listdir(new_dir) if i.startswith('train') and i.endswith('.jpg')]
    outputs.sort(key=lambda f: int(re.sub('\D', '', f)))
    outputs = [imageio.imread(new_dir + '/'+ i) for i in outputs][1:]
    imageio.mimsave(new_dir+'/annotated_results.gif', outputs, duration = 1.5)
#returns a side by side gif of images and scores through the epochs
#three different visualization settings available
#expanding starts with epoch zero and continues to add on scores through all epochs
#rolling window shows only a given window of scores which can help keep scores clearer
#static_line has all scores graphed and has a line which indicates the epoch
#title by default will be the gans settings but can be changed
#can also change duration of each image as needed
def graph_and_image_gif(directory, epochs, title = 'default', dur = 1.5, style = 'expanding', window = 5):
    with open(f'{directory}/models_scores/scores.dat', 'rb') as f:
        d_loss_fake = pickle.load(f)
        d_loss_real = pickle.load(f)
        g_loss = pickle.load(f)
    if style == 'expanding':
        new_dir = directory+'/expanding_visualization_output'
        build_directory(new_dir)
        path_list = []
        for i in range(1,epochs+1):
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,10))
            if title == 'default':
                with open(directory+'/readme.txt', 'r') as f:
                    title_1 = f.read()
                fig.suptitle(title_1)
            else:
                fig.suptitle(title)
            ax1.plot(range(i), d_loss_fake[:i], color = 'red', linewidth = 4, alpha = 0.6, label ='d_loss_fake')
            ax1.plot(range(i), d_loss_real[:i], color = 'blue', linewidth = 4, alpha = 0.5, label ='d_loss_real')
            ax1.plot(range(i), g_loss[:i], color = 'purple', linewidth = 4, alpha = 0.5, label = 'g_loss')
            ax1.legend()
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss Score')
            ax1.set_title(f'Model Scores from Epoch {1} to {i}')
            ax2.imshow(np.asarray(Image.open(f'{directory}/output/train_out_epoch_{i}.jpg')))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(f'Epoch {i} Image')
            path = new_dir+f'/ev_out_epoch_{i}.jpg'
            plt.savefig(path)
            plt.close()
            path_list.append(path)
        image_list = [imageio.imread(j)  for j in path_list]
        imageio.mimsave(new_dir+'/ex_viz.gif', image_list, duration = dur)


    if style == 'rolling_window':
        new_dir = directory+'/rolling_window_visualization_output'
        build_directory(new_dir)
        path_list = []
        for i in range(epochs+1):
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,10))
            if title == 'default':
                with open(directory+'/readme.txt', 'r') as f:
                    title_1 = f.read()
                fig.suptitle(title_1)
            else:
                fig.suptitle(title)
            if window+1>=i:
                ax1.plot(range(0, i), d_loss_fake[0:i], color = 'red', linewidth = 4, alpha = 0.6, label ='d_loss_fake')
                ax1.plot(range(0, i), d_loss_real[0:i], color = 'blue', linewidth = 4, alpha = 0.5, label ='d_loss_real')
                ax1.plot(range(0, i), g_loss[0:i], color = 'purple', linewidth = 4, alpha = 0.5, label = 'g_loss')
            else:
                ax1.plot(range(i-window, i), d_loss_fake[i-window:i], color = 'red', linewidth = 4, alpha = 0.6, label ='d_loss_fake')
                ax1.plot(range(i-window, i), d_loss_real[i-window:i], color = 'blue', linewidth = 4, alpha = 0.5, label ='d_loss_real')
                ax1.plot(range(i-window, i), g_loss[i-window:i], color = 'purple', linewidth = 4, alpha = 0.5, label = 'g_loss')
            ax1.legend()
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss Score')
            if window+1>i:
                ax1.set_title(f'Model Scores from Epoch 0 to {i}')
            else:
                ax1.set_title(f'Model Scores from Epoch {i-window+1} to {i}')
            ax2.imshow(np.asarray(Image.open(f'{directory}/output/train_out_epoch_{i}.jpg')))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(f'Epoch {i} Image')
            path = new_dir+f'/rwv_out_epoch_{i}.jpg'
            plt.savefig(path)
            plt.close()
            path_list.append(path)
        image_list = [imageio.imread(j)  for j in path_list]
        imageio.mimsave(new_dir+'/rwv.gif', image_list, duration = dur)

    if style == 'static_line':
        new_dir = directory+'/static_line_visualization_output'
        build_directory(new_dir)
        path_list = []
        for i in range(0,epochs+1):
            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,10))
            if title == 'default':
                with open(directory+'/readme.txt', 'r') as f:
                    title_1 = f.read()
                fig.suptitle(title_1)
            else:
                fig.suptitle(title)
            ax1.plot(range(epochs), d_loss_fake, color = 'red', linewidth = 4, alpha = 0.6, label ='d_loss_fake')
            ax1.plot(range(epochs), d_loss_real, color = 'blue', linewidth = 4, alpha = 0.5, label ='d_loss_real')
            ax1.plot(range(epochs), g_loss, color = 'purple', linewidth = 4, alpha = 0.5, label = 'g_loss')
            ax1.vlines(x= i, ymin = 0, ymax = max(d_loss_fake + d_loss_real+ g_loss), linestyle = '--', linewidth = 4, color = 'black')
            ax1.legend()
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss Score')
            ax1.set_title(f'Model Scores from Epoch')
            ax2.imshow(np.asarray(Image.open(f'{directory}/output/train_out_epoch_{i}.jpg')))
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title(f'Epoch {i} Image')
            path = new_dir+f'/sl_out_epoch_{i}.jpg'
            plt.savefig(path)
            plt.close()
            path_list.append(path)
        image_list = [imageio.imread(j)  for j in path_list]
        imageio.mimsave(new_dir+'/static_line.gif', image_list, duration = dur)
