import threading
import cv2
from ultralytics import YOLO
import os

# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')

# Define the video files for the trackers
video_file1 = "C:\Users\keith\OneDrive\Desktop\gp\graphicsPrograming\lab 10\traffic.mp4"  # Path to video file, 0 for webcam
video_file2 = "C:\Users\keith\OneDrive\Desktop\gp\graphicsPrograming\lab 10\traffic4.mp4"
video_file3 = "C:\Users\keith\OneDrive\Desktop\gp\graphicsPrograming\lab 10\traffic4.mp4"
