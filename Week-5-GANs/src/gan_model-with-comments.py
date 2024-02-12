import torch
import torch.nn as nn
import torch.nn.functional as F
## Try writing comments where a hash has been left
## What are the main differences between the generator and discriminator class?
## Search the pytorch reference for classes you have not seen before:
## https://pytorch.org/docs/stable/index.html

#  Here we define the generator class
#  The generator is nearly a symmetrical network to the discriminator
#  It takes a small input (latent vector z) and expands it up to an image with transposed convolutional layers
class Generator(nn.Module):
    # Constructor for the class
    # This gets called when a new instance of this class is instatiated
    def __init__(self, z_dim, n_f_maps, num_channels):
        # Call the constructor of the base class nn.module
        super(Generator, self).__init__()
        
        # The transposed convolutional layers of the network
        # There are four layers, all with a kernal size of four
        # The stride in layer 1 is 1 pixel, the stride is two in the higher layers
        # For more info on transposed convolutions see: https://medium.com/@marsxiang/convolutions-transposed-and-deconvolution-6430c358a5b6
        self.conv1 = nn.ConvTranspose2d(z_dim, n_f_maps * 4, kernel_size=4, stride=1, padding=0, bias=False)
        self.conv2 = nn.ConvTranspose2d( n_f_maps * 4, n_f_maps * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.ConvTranspose2d( n_f_maps * 2, n_f_maps, 4, 2, 1, bias=False)
        self.conv4 = nn.ConvTranspose2d( n_f_maps, num_channels, 4, 2, 1, bias=False)

        # Batch normalisation layers
        # Used for regularisation in GAN training
        self.bn1 = nn.BatchNorm2d(n_f_maps * 4)
        self.bn2 = nn.BatchNorm2d(n_f_maps * 2)
        self.bn3 = nn.BatchNorm2d(n_f_maps)

    # Definition of the forward pass
    # Here the generator takes latent variable z as input x
    def forward(self, x):
        # Pass input through first convolutional layer with relu activation function
        x = F.relu(self.conv1(x))
        # First batch norm layer
        x = self.bn1(x)
        # Pass input through second convolutional layer with relu activation function
        x = F.relu(self.conv2(x))
        # Second batch norm layer
        x = self.bn2(x)
        # Pass input through third convolutional layer with relu activation function
        x = F.relu(self.conv3(x))
        # Third batch norm layer
        x = self.bn3(x)
        # Pass input through fourth and final convolutional layer with tanh activation function
        x = F.tanh(self.conv4(x))
        # output generated image
        return x

#  Here we define the Discriminator class
#  The disciminator is a binary classifier (Fake vs Real)
#  This model is very similiar to the code in Week 3 for image classification
class Discriminator(nn.Module):
    
    # Constructor for the class
    def __init__(self, n_f_maps, num_channels):
        # Call the constructor of the base class nn.module
        super(Discriminator, self).__init__()
        
        # These are the four convolutional layers in the discriminator
        # These layers use strided convolutions instead of max-pooling to reduce the dimensionality of the data
        # They all have a kernel size of 4 and stride of 2 (except the final layer where the stride is 1)
        self.conv1 = nn.Conv2d(num_channels, n_f_maps, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_f_maps, n_f_maps * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(n_f_maps * 2, n_f_maps * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(n_f_maps * 4, 1, 4, 1, 0, bias=False)

        # Batch for layers for regularising training
        self.bn1 = nn.BatchNorm2d(n_f_maps * 2)
        self.bn2 = nn.BatchNorm2d(n_f_maps * 4)

    # Definition of the forward pass
    # Here the generator takes an image as input x
    def forward(self, x):
        # Pass input through first convolutional layer with leaky relu activation function
        x = F.leaky_relu(self.conv1(x))
        # Pass input through first convolutional layer with leaky relu activation function
        x = F.leaky_relu(self.conv2(x))
        # First batch norm layer
        x = self.bn1(x)
        # Pass input through first convolutional layer with leaky relu activation function
        x = F.leaky_relu(self.conv3(x))
        # Second batch norm layer
        x = self.bn2(x)
        # Pass input through first convolutional layer with leaky relu activation function
        x = F.sigmoid(self.conv4(x))
        # Output single variable which gives binary classification prediction (real or fake image)
        return x
    
