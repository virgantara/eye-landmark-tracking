from scipy.spatial import distance as dist
from imutils.video import FileVideoStream

from imutils import face_utils
import imutils
import time
import dlib
import cv2
import csv
from datetime import datetime


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear, A, B, C


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0

print("Loading predictor...")
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("Streaming...")

video_path = "video.mp4"
vs = FileVideoStream(video_path).start()

fileStream = True

# fileStream = False
time.sleep(1.0)

eye_blink_features = []
features_info = ['FrameNo', 'A_p2p6_L', 'B_p3p5_L', 'C_p1p4_L', 'ear_L', 'A_p2p6_R', 'B_p3p5_R', 'C_p1p4_R', 'ear_R',
                 'avg_ear']
frame_number = 0

while True:

    if fileStream and not vs.more():
        break

    waktu = str(datetime.now())

    frame = vs.read()

    frame_number += 1
    frame = imutils.resize(frame, width=1024)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        eye_left = eye_aspect_ratio(leftEye)
        eye_right = eye_aspect_ratio(rightEye)

        leftEAR = eye_left[0]
        rightEAR = eye_right[0]
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

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):

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
