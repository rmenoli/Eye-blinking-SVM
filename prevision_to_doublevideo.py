#USAGE

#python prevision_to_doublevideo.py --video video/nome_video.mp4 --prevision prevision/data_final_nome_prevision.csv



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

our_video=cv2.VideoCapture(args["video"])
fps=our_video.get(cv2.CAP_PROP_FPS)
fps=int(fps)
print("video a {} fps". format(fps))
# start the video stream thread
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

SHOWCASE_DATA=pd.read_csv(args["prevision"], index_col="frame")
SHOWCASE_DATA_CUMSUM=SHOWCASE_DATA.cumsum(axis=0)
SHOWCASE_DATA_CUMSUM=SHOWCASE_DATA_CUMSUM.drop('ear_norm', 1)

def mediaMoblieBlinkRate(df, wind):
    list_blink=list(df)
    list_blink_tmp=list()
    for i in range(wind,len(list_blink)):
        list_blink_tmp.append(sum(list_blink[i-wind:i])/wind)
    series_blink=pd.Series(list_blink_tmp, index=range(wind,len(list_blink)))
    return series_blink

def smoth_BR_moving_av(df, wind):
    indici=df.index
    list_blink=list(df)
    list_blink_tmp=list()
    for i in range(int(wind/2),len(list_blink)-int(wind/2)):
        list_blink_tmp.append(sum(list_blink[i-int(wind/2):i+int(wind/2)])/wind)
    series_blink=pd.Series(list_blink_tmp, index=range(indici[0]+int(wind/2),indici[-1]-int(wind/2)))
    return series_blink
	
DF_BLINK=SHOWCASE_DATA
#calcolo medie mobili per fps con finestre da 20 sec
DF_MOV_BR=mediaMoblieBlinkRate(list(DF_BLINK.blink), 20*fps)
#trasformo blink per frame in blink per min
DF_MOV_BR=DF_MOV_BR*fps*60
SMOOTH_BR=DF_MOV_BR.rolling(window=3*fps,center=False).mean()
DF_BLINK=DF_BLINK[DF_BLINK.blink>0]

FRAME = 1

secondi=SHOWCASE_DATA.index/fps
my_xticks=list()
for i in secondi:
    my_xticks.append(time.strftime("%M:%S", time.gmtime(i)))


while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    
    fig=plt.figure()
    ax1 = fig.add_subplot(2, 1,1)
    SHOWCASE_DATA.ear_norm[max([0,FRAME-20*fps]):FRAME+1].plot(color="blue",label='EAR norm')
    plt.plot(DF_BLINK.index ,DF_BLINK.ear_norm,'o', color="red", label='Bink')
    plt.legend(loc=2, prop={'size': 8})
    plt.ylim([0,1])
    frequency=(fps*5)
    plt.xticks(SHOWCASE_DATA.index[max([0,FRAME-20*fps]):FRAME+1:frequency], my_xticks[max([0,FRAME-20*fps]):FRAME+1:frequency] )

    ax2 = fig.add_subplot(2, 1,2)  
    SMOOTH_BR.plot(color="r", label="Blink/min")
    plt.legend(loc=2, prop={'size': 8})
    plt.axvline(x=FRAME, color="black")
    frequency=fps*30
    plt.xticks(SHOWCASE_DATA.index[::frequency], my_xticks[::frequency], rotation=45)
	
    plt.savefig('plot.png')
    plt.close()
    plot = cv2.imread('plot.png')
    plot = imutils.resize(plot, width=450)
	


    try:
	    frame=np.concatenate((frame,plot), axis=0)
	    cv2.putText(frame, "OpenCV blink detection: {}".format(SHOWCASE_DATA_CUMSUM.threshold[FRAME]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    cv2.putText(frame, "SVM    blink detection: {}".format(SHOWCASE_DATA_CUMSUM.blink[FRAME]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    try:
		    cv2.putText(frame, "Blink/min: {}".format(round(DF_MOV_BR[FRAME])), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	    except:
		    cv2.putText(frame, "Blink/min: collecting data", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    except:
	    cv2.putText(frame, "End of Data", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    try:
        cv2.imwrite("final_video/frame{}.jpg".format(FRAME), frame)
        print("image frame{}.jpg saved".format(FRAME))
    except:
        print("imwrite error")

    FRAME += 1

cv2.destroyAllWindows()
vs.stop()
