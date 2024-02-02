'''
READ BEFORE PROCEEDING:

To run python code from VS Code, in the Terminal, 
you need to press the play button on the top right corner of the UI.

The button opens a terminal panel in which your Python interpreter 
is automatically activated, and runs "python3 name.py" (macOS/Linux)
or "python name.py" (Windows).

Before running the code, you need to select your Python interpreter,
similarly to how you select your kernel for your ipynb notebooks.
From the Command Palette (⇧⌘P), select "Python: Select Interpreter"
and from there, pick your conda environment, i.e. "aim".

Press ESC to exit the Command Palette if needed.
'''

# COLLECTING IMAGES FROM WEBCAM FOR YOUR DATASET

# This script allows you to use your webcam to collect images/frames
# for 3 different classes of human poses. Feel free to collect images
# of yourself in 3 different poses.

# By the time you run the programme, a window will pop up and you will 
# be expected to press "y" to start collecting images. There is a 5 sec 
# buffer for you to strike the pose after you press "y".

# When collecting images of yourself, do not forget to provide variations 
# of each pose (angle, height, facing, proximity to the camera, e.t.c.) to
# ensure better accuracy for your classifier.

# If needed, change the number of classes, the size of each class, 
# i.e. the number of samples per class, and the buffer time (5 sec
# might not be enough if you want to move further away from the cam)
# Have fun!

import os
import cv2

# you need to be in your AI-4-MEDIA-23-24 directory
# to make sure you are in the correct directory, open the terminal and type in:
# cd PATH_TO_/AI-4-MEDIA-23-24
IMG_DIR = './data/my-data/my-pose-classification-dataset'
# define the number of classes of the different poses to collect
NUM_OF_CLASSES = 3
# define the number of samples per class
NUM_OF_FRAMES = 100

# if the path does not exist, it will create it
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# define a video capture object
cam = cv2.VideoCapture(0)

for i in range(NUM_OF_CLASSES):
    # create the folders for each class: 0, 1, 2
    if not os.path.exists(os.path.join(IMG_DIR, str(i))):
        os.makedirs(os.path.join(IMG_DIR, str(i)))

    # message to be printed in the terminal, asking the user to press "y" for data collection
    print('Press "y" to start collecting data for class: {}'.format(i+1))

    while(True): 
        # capture the video frame by frame 
        success, frame = cam.read() 

        # countdown for a buffer of 5 sec until it starts collecting frames
        if success and cv2.waitKey(25) == ord('y'):
            j = 5
            while j > 0:
                print('Start collecting in {}'.format(j))
                cv2.waitKey(1000)
                j -= 1
            break
        cv2.imshow('frame', frame)

    # collect a set number of samples for each class of the loop
    counter = 0
    print('Collecting...')
    while counter < NUM_OF_FRAMES:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(IMG_DIR, str(i), '{}.jpg'.format(counter)), frame)
        counter += 1

# message to be printed in the terminal, after all classes have been collected
print('Collected image data for {} different classes'.format(NUM_OF_CLASSES))

# release the cam object 
cam.release() 
# destroy all the windows 
cv2.destroyAllWindows() 