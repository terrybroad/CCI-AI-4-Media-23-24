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

# RUNNING THE POSE CLASSIFIER ON REAL-TIME INPUT FROM THE WEBCAM

# This script allows you to use your webcam and classify all the 
# detected human poses based on the pose classifier that we 
# created in notebook 02.

# If needed, change the number and name of classes and the model path.
# Have fun!

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from src.Dorothy import Dorothy
from src.yolo_draw_utils import draw_skeleton

# Initialize YOLO model
model = YOLO('yolov8n-pose.pt') 
dot = Dorothy()

# Define NeuralNet class for pose classification as we defined it in the training notebook 02
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(34, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

# Device, model path and classes
device = 'cpu'
model_path = './week-4-sensing-bodies/pose_classifier.pt'
classes = ['closed-arms', 'lift-leg', 'open-arms']

# Load pose classifier model
pose_classifier = NeuralNet()
pose_classifier.load_state_dict(torch.load(model_path, map_location=device))
pose_classifier.eval()

# Class for handling camera feed and drawing
class MySketch:
    camera = cv2.VideoCapture(0)

    def __init__(self):
        dot.start_loop(self.setup, self.draw)  

    def setup(self):
        print("Setup")
        # Turn off autofocus
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    def draw(self):
        success, camera_feed = self.camera.read()
        
        if success:
            # Convert color and resize to canvas size
            camera_feed = cv2.cvtColor(camera_feed, cv2.COLOR_BGR2RGB)
            camera_feed = cv2.resize(camera_feed, (dot.width, dot.height))

            # Process frame with YOLO model
            results = model(camera_feed)
            
            if results[0].keypoints != None:
                # Get skeleton keypoints from YOLO results
                poses = results[0].keypoints.data 
                pose_list = torch.split(poses,1,0)
                for pose in pose_list:
                    camera_feed = draw_skeleton(camera_feed, pose.squeeze())
                result_keypoint = results[0].keypoints.xyn.cpu().numpy()[0]

                # Return some keypoint_data even if no body is detected
                if (len(result_keypoint)) == 0:
                    keypoint_data = [0.0] * 34
                else:
                    keypoint_data = result_keypoint.flatten().tolist()
                # Turn our array into a tensor 
                if not isinstance(keypoint_data, torch.Tensor):
                    keypoint_data = torch.tensor(keypoint_data, dtype=torch.float32)

                # Perform pose classification
                prediction = pose_classifier(keypoint_data)
                _, predict = torch.max(prediction, -1)
                label_predict = classes[predict]

                # Display the pose classification result
                cv2.putText(camera_feed, f'Pose: {label_predict}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)

            dot.canvas = camera_feed

# Run the sketch
MySketch()
