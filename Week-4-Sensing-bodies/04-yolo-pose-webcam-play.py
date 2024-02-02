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

# This programme is using Dorothy, a Creative Computing Python Library for Audio Reactive Drawing
# created by Louis McCallum
# Feel free to explore it and get creative!
# https://github.com/Louismac/dorothy/tree/main

import cv2
import torch

from ultralytics import YOLO

from src.Dorothy import Dorothy
from src.yolo_draw_utils import draw_skeleton

model = YOLO('yolov8n-pose.pt') 

dot = Dorothy()

class MySketch:
    
    # Set up camera
    camera = cv2.VideoCapture(0)

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("setup")
        # Turn off autofocus
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    def draw(self):
        # Pull in frame
        success, camera_feed = self.camera.read()
        if success:
            # Convert color
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            # Resize to canvas size
            camera_feed = cv2.resize(camera_feed,(dot.width, dot.height))
            
            # Process frame with YOLO model
            results = model(camera_feed)
            
            if results[0].keypoints != None:
                # Get skeleton keypoints from YOLO results
                poses = results[0].keypoints.data
                pose_list = torch.split(poses,1,0)
                
                for pose in pose_list:
                    camera_feed = draw_skeleton(camera_feed, pose.squeeze())
   
            dot.canvas = camera_feed
             
MySketch()  





