#USAGE

#python show_video_EAR.py --shape-predictor shape_predictor_68_face_landmarks.dat --video video/nome_video.avi --prevision prevision/nome_prevision.csv



# import the necessary packages
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import pandas as pd
import argparse
import imutils
import time
import dlib
import cv2

#fetch arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
ap.add_argument("-q", "--prevision", type=str, default="",
                help="path to prevision file csv")
args = vars(ap.parse_args())

#facial detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# start the video stream thread
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

SHOWCASE_DATA=pd.read_csv(args["prevision"], index_col="frame")
SHOWCASE_DATA_CUMSUM=SHOWCASE_DATA.cumsum(axis=0)
SHOWCASE_DATA_CUMSUM=SHOWCASE_DATA_CUMSUM.drop('ear_norm', 1)

DF_BLINK=SHOWCASE_DATA
DF_BLINK=DF_BLINK[DF_BLINK.blink>0]
FRAME = 0

while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    plot = SHOWCASE_DATA.ear_norm.plot(color="blue")
    plt.plot(DF_BLINK.index ,DF_BLINK.ear_norm,'o', color="red")
    plt.axvline(x=FRAME, color="black")
    plt.savefig('plot.png')
    plot = cv2.imread('plot.png')
    plot = imutils.resize(plot, width=450)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        try:
            frame=np.concatenate((frame,plot), axis=0)
            cv2.putText(frame, "OpenCV blink detection: {}".format(SHOWCASE_DATA_CUMSUM.threshold[FRAME]), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Frame: {}".format(FRAME), (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "SVM    blink detection: {}".format(SHOWCASE_DATA_CUMSUM.blink[FRAME]), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except:
            cv2.putText(frame, "End of Data", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    try:
        cv2.imwrite("final_video/frame{}.jpg".format(FRAME), frame)
        print("image frame{}.jpg saved".format(FRAME))
    except:
        print("imwrite error")
    FRAME += 1

cv2.destroyAllWindows()
vs.stop()
