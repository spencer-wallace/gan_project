# Artwork Generation Using Generative Adversarial Networks and Met Images #
## Introduction  ##
Since first being proposed by Goodfellow et al, Generative Adversarial Networks(GANs) have occupied a significant place in the debate surrounding AI. These networks have drawn interest from academic spheres as well as the general public. Academic interest has led to additional diversification of and improvements on Goodfellow et al's concept with variations such as style gans and info gans growing increasingly popular. These GANs have found their way into the public discourse as they power the popular app FaceApp, are behind thispersondoesnotexist.com, and are frequently used in deepfakes. GANs seem poised to continue playing a role in the developing AI landscape as they continue to improve and expand in their capabilities.

While GANs continue to grow in popularity, there are still no widely available packages that help make them more accessible. If one wants to make a GAN, currently they must build one from scratch which oftentimes means tailoring and editing an existing example to their needs. On many levels, this can be viewed as a positive as building these models from the ground up can help build a level of understanding of how they operate. On the other hand, building from the ground up means that a good deal of work needs to be done to reconfigure the networks to experiment with different configurations and parameters. This makes experimentation more difficult and makes working with GANs more difficult than is necesary. 

Information about the dataset used can be found [here](https://metmuseum.github.io/) and [here](https://github.com/metmuseum/openaccess).
## Aims ##
The aims of this project can be separated into two broad categories: specific and general.

The specific aims relate directly to generating images from collected data. These aims look much more like a traditional data science workflow than the general aims and are as follows:
1. Collect data using web scraping and API calls
2. Clean data
3. Store data effectively
4. Build and train GAN
5. Generate novel images
6. Present the novel images

The general aims relate to how what has been learned and done at the specific level can be generalized to other contexts. Concretely, this meant developing a function library that allowed for rapid reconfiguring of the GANs structure and parameters along with a unified method of storing resulting data and some tools to help with visualisations. The ultimate goal for this library was to allow for rapid construction of a number of GANs in order to better understand which settings improve the results. In this case, this took the shape of setting a hyperparameter space(much like sklearn's GridSearchCV) which is used to create a number of possible GAN configurations which are then run. 

The general and specific level aims dovetail nicely as the specific aims shape the general aims and the general aims help simplify and smooth the process of the specific aims. 

## Project  Overview ##
### Data Collection, Preparation and Storage ###
Utilizing a combination of API calls and web scraping, 500,000 images and the corresponding data of artwork and artifacts were collected from the New York Metropolitan Muesuem of Art's online collections. This required an initial API call which brought back a list of the object IDs of all the work in the collections. This list could then be used in a second API call in order to acquire the links for the images which then allowed for a web scrape to obtain the images. The data and images were then cleaned and stored. The data relating to the image was stored in PostgreSQL while the images were stored on an external disk using their image ID as a way to connect to their row in the SQL table. In order to make the GAN work as well as possible, a subset of these images needed to be selected for a number of reasons:
1. The dataset is far too large from a computational perspective
2. The dataset is varied, with a mixture of artifacts and works of art from across human history. This means many of the images have littel to nothing in common. This makes training a GAN very difficult. 
3. A dataset of the right size needs to be large enough to allow the GAN to be trained to a high level, but also small enough that the training can be done on a reasonable timeframe. Given that the this was my first contact with GANs, the decision was made to err on the side of smaller rather than larger in order to have more opportunities to try different hyperparameter combinations at the possible expense of slightly undertrained GANs.

Finding the correct subset of items required a good deal of familiarity with the dataset and careful consideration. In the end, the dataset's roughy 5000 vases were chosen as the dataset because:
1. At roughly 5,000 images, they represented a happy medium for dataset size.
2. While there is still a good amount of variance in the images, the vases represent a fairly clean dataset. The images are all presented on an off-white background and the objects themselves share similar form factors.
3. A final consideration is that most of the images have a very similar height to width ratio. This meant that it was easy to process the photos to the same size without having to worry about the impact that compressing the images into a very different shape may have on the outputs. The same can not be same for many of the other object types in the collection where there is a wide variety of ratios. 

The entire process of collecting, cleaning and storing the data was documented in a set of Jupyter Notebooks.

The two intial API calls: [1.acquire_image_urls](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/1.acquire_image_urls.ipynb)
The web scrape and data storage: [2.images_postgreSQL_and_scrape](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/2.images_postgreSQL_and_scrape.ipynb)
Cleaning and further organization of the data: [3.clean_and_organize_data](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/3.clean_and_organize_data.ipynb)
Selecting the vases and additional cleaning/perparation: [4.select_and_clean_vase_data](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/4.select_and_clean_vase_data.ipynb)

Finally, a copy of the dataset is provided: vase_image_data.zip
Please note that due to size limitations for files from GitHub, the files found in the zipped folder are smaller than those I used on my local machine. If you would like the higher resolution images, you can use the notebooks listed aboce to acquire the data. 

## Getting Started ##
Note: The project was developed using Python 3 on an Anaconda distribution. Installation instructions can be found [here](https://docs.anaconda.com/anaconda/install/).
To get started, first install the required libraries inside a virtual environment:

<pre><code>pip install -r requirements.txt
</code></pre>

If planning to use the included data, you must unzip the file, which can be done in the command line using:

<pre><code>unzip vase_image_data.zip
</code></pre>

If collecting your own data from the met using the code provided, PostgreSQL is needed. Instructions on downloading Postgres can be found [here](https://www.postgresql.org/download/) To start, open the jupyter notebook titled [1.acquire_image_urls](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/1.acquire_image_urls.ipynb) and follow the steps laid out there, following through with [Notebook 2](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/2.images_postgreSQL_and_scrape.ipynb) and [Notebook 3](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/3.clean_and_organize_data.ipynb). Any additional data prep can be modelled on [Notebook 4](https://github.com/spencer-wallace/gan_project/blob/main/Data%20Collection%20and%20Cleaning/4.select_and_clean_vase_data.ipynb), but it is worth noting that it is specific to the vases collection and may need to modified for your use case.

To run a GAN, start by opening [gan_search](https://github.com/spencer-wallace/gan_project/blob/main/gan_search.py) in a text editor and set the hyper parameter space as desired. Please note that it is best to start with one setting for each parameter if this is your first time working with GANs. I would recommend starting with small x and y sizes to allow for quicker experimentation. Additionally, I would recommend keeping run_checkpoint set to 1. It provides a pair of callbacks which will show an example image result after each epoch to allow for easier tracking of how the network is training and a second callback which will stop the models training and move on if the model converges to generating images that are a single color for a set number of epochs. Once you have established the desired parameter space, you can ran the script from your command line by entering:

<pre><code>python3 gan_search
</code></pre>

The gan_library also contains a number of tools within the viz_tools file that can help visualize the progression of a GAN. There are tools to create a GIF of each epoch's generated image anotated with its scores and settings as well as a tool to create a GIF of the image alongside a graph of scores with several settings for how the graph is displayed.
 
## Dependencies ##
Please see [requirements.txt](https://github.com/spencer-wallace/gan_project/blob/main/requirements.txt)

## License ##
The data was provided by the New York Metropolitan Museum where it is held under a [Creative Commons Zero v1.0 Universal License](https://github.com/metmuseum/openaccess/blob/master/LICENSE)

This project is licensed with a [Creative Commons Zero v1.0 Universal License](https://creativecommons.org/publicdomain/zero/1.0/legalcode)

