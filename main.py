from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd
import csv
from datetime import datetime

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear, A, B, C


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="",
# 	help="path to input video file")
# args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")

video_path = "video.mp4"
vs = FileVideoStream(video_path).start()
# vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

eye_blink_features = []
features_info = ['FrameNo', 'A_p2p6_L', 'B_p3p5_L', 'C_p1p4_L', 'ear_L', 'A_p2p6_R', 'B_p3p5_R', 'C_p1p4_R', 'ear_R',
                 'avg_ear']
# loop over frames from the video stream
frame_number = 0

while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    waktu = str(datetime.now())
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()

    frame_number += 1
    frame = imutils.resize(frame, width=1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        eye_left = eye_aspect_ratio(leftEye)
        eye_right = eye_aspect_ratio(rightEye)

        leftEAR = eye_left[0]
        rightEAR = eye_right[0]
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        ft = {
            'FrameNo': frame_number,
            'A_p2p6_L': eye_left[1],
            'B_p3p5_L': eye_left[2],
            'C_p1p4_L': eye_left[3],
            'ear_L': leftEAR,
            'A_p2p6_R': eye_right[1],
            'B_p3p5_R': eye_right[2],
            'C_p1p4_R': eye_right[3],
            'ear_R': rightEAR,
            'avg_ear': ear
        }

        # print(ft)
        eye_blink_features.append(ft)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        # if ear < EYE_AR_THRESH:
        # 	COUNTER += 1
        # # otherwise, the eye aspect ratio is not below the blink
        # # threshold
        # else:
        # 	# if the eyes were closed for a sufficient number of
        # 	# then increment the total number of blinks
        # 	if COUNTER >= EYE_AR_CONSEC_FRAMES:
        # 		TOTAL += 1
        # 	# reset the eye frame counter
        # 	COUNTER = 0
        #
        # 	# draw the total number of blinks on the frame along with
        # 	# the computed eye aspect ratio for the frame
        # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
        # 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        # hist_df = pd.DataFrame(eye_blink_features)
        # hist_csv_file = 'fitur_eye_blink.csv'
        # with open(hist_csv_file, mode='w') as f:
        #     hist_df.to_csv(f)
        in_fnam = 'tmp.csv'
        out_fnam = 'fitur_mata.csv'
        with open(in_fnam, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=features_info)
            print(eye_blink_features)
            writer.writeheader()
            writer.writerows(eye_blink_features)

        with open(in_fnam, newline='') as in_file:
            with open(out_fnam, 'w', newline='') as out_file:
                writer = csv.writer(out_file)
                for row in csv.reader(in_file):
                    if row:
                        writer.writerow(row)
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
