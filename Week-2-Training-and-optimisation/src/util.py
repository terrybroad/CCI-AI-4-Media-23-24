import os
import imageio
import numpy as np

# Code adapted from: https://stackoverflow.com/a/45258744
def make_training_gif(im_folder_path, im_ext, file_out):
    images = []

    # Go trough all files in folder
    for file_name in sorted(os.listdir(im_folder_path)):
        if file_name.endswith(im_ext):
            file_path = os.path.join(im_folder_path, file_name)
            images.append(imageio.imread(file_path))

    # Make it pause at the end
    for _ in range(10):
        images.append(imageio.imread(file_path))

    # Save training gif
    imageio.mimsave(file_out, images)


# Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
def get_normalised_coordinate_grid(image_shape):
    width = np.linspace(-1, 1, image_shape[0])
    height = np.linspace(-1, 1, image_shape[1])
    mgrid = np.stack(np.meshgrid(width, height), axis=-1)
    mgrid = np.reshape(mgrid, [-1, 2])
    return mgrid