#USAGE

#python show_video_EAR_NOFACE.py --video video/nome_video.avi --prevision prevision/nome_prevision.csv



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
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
ap.add_argument("-q", "--prevision", type=str, default="",
                help="path to prevision file csv")
args = vars(ap.parse_args())


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
    plot = SHOWCASE_DATA.ear_norm.plot(color="blue")
    plt.plot(DF_BLINK.index ,DF_BLINK.ear_norm,'o', color="red")
    plt.axvline(x=FRAME, color="black")
    plt.savefig('plot.png')
    plot = cv2.imread('plot.png')
    plot = imutils.resize(plot, width=450)
    

    try:
	    frame=np.concatenate((frame,plot), axis=0)
	    cv2.putText(frame, "OpenCV blink detection: {}".format(SHOWCASE_DATA_CUMSUM.threshold[FRAME]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    cv2.putText(frame, "Frame: {}".format(FRAME), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    cv2.putText(frame, "SVM    blink detection: {}".format(SHOWCASE_DATA_CUMSUM.blink[FRAME]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except:
	    cv2.putText(frame, "End of Data", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    try:
        cv2.imwrite("final_video/frame{}.jpg".format(FRAME), frame)
        # print("image frame{}.jpg saved".format(FRAME))
    except:
        print("imwrite error")
    FRAME += 1

cv2.destroyAllWindows()
vs.stop()
