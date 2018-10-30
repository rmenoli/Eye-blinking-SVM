# USAGE
# python video_to_plot.py --shape-predictor shape_predictor_68_face_landmarks.dat --video video/nome_video.avi
# python SVM_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
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
import sys
#########################
import numpy
from pandas import DataFrame
from matplotlib import pyplot
from pandas import read_csv
#from pandas import to_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
#########################################
import copy
#######################################
import time
import matplotlib.pyplot as plt
########################################

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
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.30
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
# vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
# fileStream = False
time.sleep(1.0)

FRAME=0

ear_list=list()

array_blink_threshold=list()

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=900)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# no face detected
	if(len(rects)==0):
		ear_list.append(np.nan)
		array_blink_threshold.append(np.nan)

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
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0
		ear_list.append(ear)
		array_blink_threshold.append(0)
		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
				array_blink_threshold[FRAME]=1
			# reset the eye frame counter
			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Frame: {}".format(FRAME), (10, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	# show the frame
	# cv2.imshow("Frame", frame)
	sys.stdout.write('\r{}'.format(FRAME))
	key = cv2.waitKey(1) & 0xFF

	FRAME += 1
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

#moving avareage function
def moving_av(mylist, N):
	cumsum, moving_aves = [0], []
	for i, x in enumerate(mylist, 1):
		cumsum.append(cumsum[i-1] + x)
		if i>=N:
			moving_ave = (cumsum[i] - cumsum[i-N])/N
			#can do stuff with moving_ave here
			moving_aves.append(moving_ave)
	return moving_aves

# try:
# 	users_final = pd.read_csv("tag/{}.tag".format(args["video"][6:-4]), sep='\t', header=None,
# 	                          names=['frame', 'tag'], index_col="frame")
# except FileNotFoundError:
# 	users_final = pd.read_csv("tag/{}.tag".format(args["video"][7:-4]), sep='\t', header=None,
# 	                          names=['frame', 'tag'], index_col="frame")

mov_ear_3=moving_av(ear_list,3)
mov_ear_5=moving_av(ear_list,5)
mov_ear_7=moving_av(ear_list,7)

ear_list = pd.Series(ear_list, index=range(0, len(ear_list)))
array_blink_threshold=pd.Series(array_blink_threshold,index=range(0, len(array_blink_threshold)))

mov_ear_3=pd.Series(mov_ear_3, index=range(2, len(mov_ear_3)+2))
mov_ear_5=pd.Series(mov_ear_5, index=range(3, len(mov_ear_5)+3))
mov_ear_7=pd.Series(mov_ear_7, index=range(4, len(mov_ear_7)+4))

ear_list = pd.DataFrame(ear_list)
ear_list["threshold"] = array_blink_threshold
ear_list["mov_ear_3"] = mov_ear_3
ear_list["mov_ear_5"] = mov_ear_5
ear_list["mov_ear_7"] = mov_ear_7
ear_list.columns = ["ear", "threshold", "mov_ear_3","mov_ear_5","mov_ear_7"]
#ear_list = ear_list.fillna(0)
#mask = ear_list.tag == 0
#ear_list.tag = ear_list.tag.where(mask, 1)

ear_list.index.name="frame"
'''
try:
	ear_list.to_csv("non_training_data_raw_data/{}/{}.csv".format(
			args["video"][6:-4]), index=True, header=True)
except FileNotFoundError:
	ear_list.to_csv("non_training_data_raw_data/{}.csv".format(
            args["video"][7:-4]), index=True, header=True)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
'''
ear_list.to_csv("tmp.csv",index=True, header=True)
##########################################################
######################################################
#unione detect_blinks con preproc_svm
#####################################################
######################################################
dati=ear_list
#######################################################
listear=list(dati.ear)

#normalizzo
listear=np.array(listear)
listear=(listear-np.nanmin(listear))/(np.nanmax(listear)-np.nanmin(listear))
listear=list(listear)
LIST_EAR_PER_TABELLA_PREVISIONI=listear
LIST_EAR_PER_TABELLA_PREVISIONI=pd.Series(LIST_EAR_PER_TABELLA_PREVISIONI, index=range(0,len(LIST_EAR_PER_TABELLA_PREVISIONI)))

col=['F1',"F2","F3","F4","F5",'F6',"F7"]
df_fin=pd.DataFrame(columns=col)

#creo righe da 9 frame
for i in range(3, len(listear)-4):
    tmp_ear=listear[i-3:i+4]
    df_fin.loc[i]=tmp_ear
	
df_fin.index.name="frame"
df_fin.dropna(how='any', inplace=True)

#####################################################################
####################################################################
#unione preproc_svm con ML
#############################################################à
######################################################à
dataset=pd.read_csv("./balanced_preproc/balanced_preproc_all.csv", index_col="frame")
# Split-out validation dataset
array = dataset.values
X = array[:,:dataset.shape[1]-1].astype(float)
Y = array[:,dataset.shape[1]-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'


# prepare the model
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=1.7)  #choose our best model and C
model.fit(rescaledX, Y_train)

# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print(roc_auc_score(Y_validation,predictions))

def prev_to_csv(X,scaler=scaler,model=model):
    rescaledX = scaler.transform(X)
    predictions = model.predict(rescaledX)
    newdata = DataFrame(predictions, index=X.index, columns=["blink"])
    return newdata

previsioni=prev_to_csv(df_fin)
try:
	previsioni.to_csv(
		"prevision/output_SVM_{}.csv".format(args["video"][6:-4]),index=True, header=True)
except:
	previsioni.to_csv(
		"prevision/output_SVM_{}.csv".format(args["video"][8:-4]), index=True, header=True)

##########################################################################
##########################################################################
#unisco ML con blink_ajust
####################################################################
#####################################################################à
DATA = previsioni
################################################
FRAME_LIST = list(DATA.index)
BLINK_LIST = list(DATA.blink)

#sostituisco 0.0 o 1.0 sparsi
for n in range(len(BLINK_LIST)):
    #trovo il primo 1.0
    if BLINK_LIST[n]==1.0:
        i = copy.deepcopy(n)
        #correggi 1.0 isolati: se è un 1.0 singolo (o doppio) diventa 0.0 (o 0.0 0.0)
        if sum(BLINK_LIST[i:i+6])<3.0:
            BLINK_LIST[i]=0.0
        else:
            #correggi 0.0 isolati: se ci sono 0.0 singoli (o doppi) (o tripli) diventano 1.0 (o 1.0 1.0) (o 1.0 1.0 1.0)
            while (sum(BLINK_LIST[i:i+6])>=3.0):
                BLINK_LIST[i+1]=1.0
                BLINK_LIST[i+2]=1.0
                i+=1

#ora costruisco singoli 1.0 corrispondenti al blink
for n in range(len(BLINK_LIST)):
    #trovo il primo 1.0
    if BLINK_LIST[n]==1.0:
        i = copy.deepcopy(n)
        while (BLINK_LIST[i+1]==1.0):
            BLINK_LIST[i+1]=0.0
            i+=1

#scala gli 1.0 di 5 frame per posizionarlo alla chiusura circa
BLINK_LIST=[0.0,0.0,0.0,0.0,0.0]+BLINK_LIST[:len(BLINK_LIST)-5]


BLINK_LIST = pd.DataFrame(BLINK_LIST, index=FRAME_LIST)
BLINK_LIST.index.name='frame'
BLINK_LIST.columns = ['blink']

#########################################################################################
########################################################################################
#unisco blink_ajust con video_shocase
######################################################################################
###################################################################################
result=BLINK_LIST
######################################################################

our_video=cv2.VideoCapture(args["video"])
fps=our_video.get(cv2.CAP_PROP_FPS)
fps=int(fps)
print("video a {} fps". format(fps))
# start the video stream thread
vs = FileVideoStream(args["video"]).start()
fileStream = True
time.sleep(1.0)

raw_data=pd.read_csv("tmp.csv", index_col="frame")
raw_data_1=raw_data.threshold
SHOWCASE_DATA=pd.concat([raw_data_1, result,LIST_EAR_PER_TABELLA_PREVISIONI], axis=1 )
SHOWCASE_DATA=SHOWCASE_DATA.fillna(0)
SHOWCASE_DATA.columns=["threshold","blink","ear_norm"]


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

SMOOTH_BR.plot(color="r", label="Blink/min")
plt.legend(loc=2, prop={'size': 8})
plt.axvline(x=FRAME, color="black")
frequency=fps*30
plt.xticks(SHOWCASE_DATA.index[::frequency], my_xticks[::frequency], rotation=45)
plt.savefig('plot.png')
plt.close()

cv2.destroyAllWindows()
vs.stop()
