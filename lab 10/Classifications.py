import threading
import cv2
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run_tracker_in_thread(filename, model, file_index):
    """
        Press 'q' to quit the video display window.
    """
    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        cv2.imshow(f"Tracking_Stream_{file_index}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')

# Define the video files for the trackers
video_file1 = "C:/Users/keith/OneDrive/Desktop/gp/graphicsPrograming/lab 10/traffic.mp4"  # Path to video file, 0 for webcam
video_file2 = "C:/Users/keith/OneDrive/Desktop/gp/graphicsPrograming/lab 10/traffic2.mp4"
video_file3 = "C:/Users/keith/OneDrive/Desktop/gp/graphicsPrograming/lab 10/traffic3.mp4"
video_file4 = "C:/Users/keith/OneDrive/Desktop/gp/graphicsPrograming/lab 10/ownVid.mp4"

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)
tracker_thread3 = threading.Thread(target=run_tracker_in_thread, args=(video_file3, model2, 3), daemon=True)
tracker_thread4 = threading.Thread(target=run_tracker_in_thread, args=(video_file4, model1, 4), daemon=True)


# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()
tracker_thread3.start()
tracker_thread4.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()
tracker_thread3.join()
tracker_thread4.join()

# Clean up and close windows
cv2.destroyAllWindows()