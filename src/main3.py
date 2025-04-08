import numpy as np
import argparse
from camera_functions import start_camera, resize_frame, detect_faces
from pomodoro_functions import play_alarm, stop_alarm, spent
import time
import cv2

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--window", type=bool, default=False, help="Camera Window")
args = vars(ap.parse_args())

# Load the pre-trained face detection model
net = cv2.dnn.readNetFromCaffe("data/deploy.prototxt.txt", "data/res10_300x300_ssd_iter_140000.caffemodel")

# Start the camera
vs = start_camera()
time.sleep(2.0)

# Initialize timer and state variables
round_completed = False
start_time = int(time.time())
timer_reset_start = 0

# Timer values
PomodoroTime = 200  # 200 seconds for Pomodoro
IntervalTime = 60  # 60 seconds for Interval

# Capture window function
def capture_window():
    while True:
        frame = vs.read()
        frame = resize_frame(frame)
        detections, (h, w) = detect_faces(frame, net)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()
    vs.stop()

# Function to check availability based on face detection
def availability():
    total_confidence = 0
    num_faces = 0  # Track number of faces with confidence > 0.7

    for _ in range(5):  # Check 5 frames
        frame = vs.read()
        frame = resize_frame(frame)
        detections, _ = detect_faces(frame, net)

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold
                total_confidence += confidence
                num_faces += 1
        
        time.sleep(2)  # Wait for 2 seconds before checking the next frame

    # Take the average confidence
    avg_confidence = total_confidence / num_faces if num_faces > 0 else 0
    print(f"Total Confidence: {total_confidence}, Average Confidence: {avg_confidence}, Faces Detected: {num_faces}")
    
    if avg_confidence > 0.75 and num_faces > 0:
        print("Available")
        return 1
    else:
        print("Not Available")
        return 0

# Main loop
if args["window"]:
    capture_window()
else:
    while True:
        available = availability()

        if not round_completed and spent(PomodoroTime, start_time):
            round_completed = True
            start_time = int(time.time())
            play_alarm("data/Pomodoro.mp3")
            print("Round Completed")

        if round_completed and not available:
            stop_alarm()

        if round_completed and spent(IntervalTime, start_time):
            round_completed = False
            start_time = int(time.time())
            play_alarm("data/IntervalFinished.mp3")
            print("Round Again")

        if not round_completed and available:
            stop_alarm()
            timer_reset_start = 0

        if not round_completed and not available:
            if not timer_reset_start:
                timer_reset_start = int(time.time())
                print("Resetting Timer Started")
            elif int(time.time()) - timer_reset_start > IntervalTime:
                start_time = int(time.time())
                print("Timer Reset!")

        time.sleep(10)  # Sleep to avoid constant checking
