import keras
#create discriminator net as class
class Discriminator():
    #establish setting necessary to build net
    def __init__(self, y_input = 15, dense_layers = None, dense_size = None, x_input=10, n_filters=32,
                n_convols =2, relu_alpha = 0.1, rgb = 1,kernel_size = 4, strides = 2):
        #vertical length of corresponding generator inputs
        self.y_input = y_input
        #horizontal length of corresponding generator inputs
        self.x_input = x_input
        #number of filters for first convolutional layer
        self.n_filters = n_filters
        #number of convolutional layers
        self.n_convols = n_convols
        #activation alpha
        self.relu_alpha = relu_alpha
        #if images will be color or not. 0 = grayscale, 1 = color
        self.rgb = rgb
        #size of kernel for transpose convolutional layer
        self.kernel_size = kernel_size
        #stride of transpose convolutional layer
        self.strides = strides
        #number of dense layers if any
        self.dense_layers = dense_layers
        #number of neurons per dense layer
        self.dense_size = dense_size

    #build discriminator
    def build(self):
        #input shape, depending on rgb value. height and width are determined by multiplying x and y inputs by 2 to the number of convolutional layers
        if self.rgb ==1:
            self.input_shape=(self.y_input * 2**self.n_convols, self.x_input* 2**self.n_convols, 3)
        elif self.rgb == 0:
            self.input_shape=(self.y_input * 2**self.n_convols, self.x_input* 2**self.n_convols,1)
        #sequential model
        self.model = keras.models.Sequential()
        #input layer using input shape
        self.model.add(keras.layers.Input(shape=self.input_shape))
        #number of filters for first layer. Doubles each layer
        self.nf = self.n_filters
        #first convolutional block
        self.model.add(keras.layers.Conv2D(self.nf, kernel_size=(self.kernel_size, self.kernel_size),
                                            strides=(self.strides, self.strides), padding="same", use_bias=True))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.LeakyReLU(alpha=self.relu_alpha))
        # Number of filters doubled after each convolutional layer
        self.nf = self.nf*2
        # Additionaly convolutional layers past one added with for loop
        for u in range(2, self.n_convols+1):
            self.model.add(keras.layers.Conv2D(self.nf, kernel_size=(self.kernel_size, self.kernel_size),
                                                strides=(self.strides, self.strides), padding="same", use_bias=False))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.LeakyReLU(alpha=self.relu_alpha))
            self.nf = self.nf*2
        #flatten output of convolutional layers
        self.model.add(keras.layers.Flatten())
        #add dense layers if specified, else pass
        if self.dense_layers == None:
            pass

        else:
            for a  in range(self.dense_layers):
                self.model.add(keras.layers.Dense(dense_size, use_bias=False))
                self.model.add(keras.layers.BatchNormalization())
                self.model.add(keras.layers.LeakyReLU(alpha=0.1))
        # output, fake or real binary classification using sigmoid activation
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

        return self.model
