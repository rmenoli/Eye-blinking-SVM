# Intro
This repo contians the code used to develop a blink detector based on an SVM model.
Two examples of software's the output are available in the following video:
 - First example: https://www.youtube.com/watch?v=6aDerndtoUc&t=64s
 - Second example: https://www.youtube.com/watch?v=jd10kPrJ1sw&t=63s
 

# Analysis of Lying through an Eye Blink Detection Algorithm
Lie detection using physiological methods has been utilized for the past 75 years, starting with the work of Benussi, with varying rates of success in identification of the guilty.
More recently, eye blink measures have attracted considerable attention, since they have been related to cognitive processes. They can be recorded easily without the subject’s awareness and without the application of electrodes.

1. The first part of our project consisted of acquiring the background knowledge in the domain of cognitive and behavioral sciences with a focus on the relevant researches on eye blink and lie detection.

2. The second part was more technical and regarded the research of the free tools already available for eye blink detection with a good performance in terms of accuracy. To our knowledge, the only freely-available software for blink detection is built on the OpenCV Python modules and is based on a metric called eye aspect ratio (EAR). Despite the acceptable results in live blink detection, this ready-solution performed poorly in different video conditions.

3. Therefore, the third part of the project consisted of building, testing and validating a Support Vector Machine (SVM) classifier using the EAR features.

4. Finally, we created an interface for the video to analyse from a lie detection point of view, together with some useful descriptive plots of blinking measures.

## OpenCV and Facial Landmarks
The implementation which follows is based on the tutorial from the blog Pyimagesearch (see Rosebrock) which, in turn, takes as its starting point the work of Soukupova and Cech for the the metric called eye aspect ratio (EAR). The 68 facial landmarks are based on the iBUG 300-W dataset (see Sagonas et al. and figure below on the left), which the dlib facial landmark predictor was trained on.

### Eye Aspect Ratio (EAR)
For every video frame, the eye landmarks (landmarks [37,42] and [43,48] for left and rigth eye respectively) are detected. The eye aspect ratio (EAR) between height and width of the eye is computed, following the work of Soukupova and Cech.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/facial_landmarks_68markup.jpg" width="350"/> <img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/blink_detection_6_landmarks.jpg" width="350"/>
</p>

### Blink Detector using SVM
It generally does not hold that low value of the EAR means that a person is blinking. A low value of the EAR may occur when a subject closes his/her eyes intentionally for a longer time or performs a facial expression, yawning, etc., or the EAR captures a short random fluctuation of the landmarks.

Therefore, following what has been proposed by Soukupova and Cech, we built a classifier that takes as an input the EAR values belonging to a larger temporal window of a frame.

### Training
See the attached file "Report_final.pdf" to see the details.

### Validation on different video conditions
The SVM blink detector was firstly validated on some videos from the iBUG 300-W dataset and its performance was therefore compared with the OpenCV blink detector (see Rosebrock). The aim of this stage was to test the two system’s blink detection ability, independently of variations in pose, expression, illumination, background, occlusion, and image quality. 

Following what has been done in Sagonas et al., the following three scenarios have been considered:
- Scenario 1 (Top): A number of testing videos will be of people recorded in well-lit conditions displaying arbitrary expressions in various head poses. This scenario aims to evaluate algorithms that could be suitable for facial motion analysis in laboratory and naturalistic well-lit conditions.
- Scenario 2 (Middle): A number of testing videos will be of people recorded in unconstrained conditions, displaying arbitrary expressions in various head poses but without large occlusions. This scenario aims to evaluate algorithms that could be suitable for facial motion analysis in real-world human-computer interaction applications.
- Scenario 3 (Bottom): A number of testing videos will be of people recorded in completely unconstrained conditions including the illumination conditions, occlusions, makeup, expression, head pose, etc. This scenario aims to assess the performance of facial landmark tracking in arbitrary conditions.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/Validation_Scenarios.png" width="500"/>
</p>

### Validation on self-produced videos
Secondly, the validation procedure was done also for videos of friends, relatives and acquaintances; the results for two videos are summarized below. We can notice that also for new and longer videos (the lengths are 3:45 and 7:30 minutes, respectively for Bianca at the top and Filippo at the bottom) the validation procedure shows a significant improvement in the SVM blink detector in terms of both precision and recall, especially when occlusions occur (i.e. Filippo wears glasses). Moreover, the overall SVM classifier performance is better in Filippo, due to the frame rate the video was recorded at: Bianca was recorded at 14 fps, while Filippo at 29 fps. The SVM blink detector was trained with 30 fps videos, so this difference in performances was expected, but unavoidable.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/Validation_Experiments.png" width="500"/>
</p>

Bianca's output video is available at: https://www.youtube.com/watch?v=6aDerndtoUc
Filippo's output video is available at: https://www.youtube.com/watch?v=jd10kPrJ1sw&t=63s
## Is the Person Lying?
In order to determine whether there could be some empirical measures (e.g. blink rate and EAR) which could be helpful for lie detection, at first, we built an interface containing the video, some descriptive statistics and two real-time plots. 

Secondly, we decided an experimental setting that could significantly highlight differences in those empirical measures among different periods (i.e. baseline, target period and target offset), taking the work of Leal and Vrij as reference.

### Video Interface
Given a video as an input, firstly the program performs a pre-processing of the raw-data (i.e. for each frame, it detects facial landmarks, computes and normalizes the EAR values and arranges data in the form of table 1), secondly the already-trained SVM classifier computes the previsions (0 = opened eye, 1 = closed eye), then the sequence of 0’s and 1’s is converted into blinks / no blinks (see "Report_final.pdf" for more details), and finally the program returns, for each frame, an output like those showed below.

On the top, there is a real-time counter of blinks detected by the OpenCV and the SVM detectors up to that frame. Lower, it is shown the instant blink rate, which for frame x at instant t (in seconds) is computed considering the frames belonging to the interval (t-20; t).

This 20 seconds time window is then plotted in the graph below, together with the normalized EAR value (blue line) for each frame belonging to the window; a red dot is shown corresponding to the frame at which the blink was detected.

The last graph keeps track of the blink rate throughout the whole video and denotes the current frame by a vertical black line. This feature is very useful when looking at these empirical measures from a lie detection point of view.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/Interface.png" width="750"/>
</p>

### Experimental Setting
The structure of the recorded videos is organized in 5 periods of the same length, where the person follows the instructions reported on the screen:
1. Baseline: the subject is at rest, the instruction is “Look at the screen”.
2. Target Period: the subject is telling the truth, the istruction is “We ask you to tell in detail what you did last week”.
3. Baseline: the subject is at rest, the instruction is “Look at the screen”.
4. Target Period: the subject is lying, the instruction is “We ask you to tell in detail what you did last week, telling a lie (without any truth)”.
5. Baseline: the subject is at rest, the instruction is “Look at the screen”.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/Experiment.png" width="750"/>
</p>

It was observed a significant difference both in the blink rate and in the average inter-blink interval during the 5 periods. The blink rate during both target periods is higher than in the baseline periods, due to the fact that the mean blink rate at rest is 17 blinks/minute, while during conversation it increases to 26 blinks/minute (see Bentivoglio et al.). 

However, for both videos there is a significant difference in the blink rate while the person is lying and while he/she is telling the truth, independently from the order of the target periods: not only the mean blink rate is lower while the person is lying, but also the mean inter-blink interval is lower when the subject is telling the truth.

## Conclusions
1. This work has firstly developed a robust and efficient blink detector, based on the eye aspect ratio (EAR) value given as input to a fully-trained and validated SVM machine learning model. The performance was much higher than the existing OpenCV blink detector, under many video qualities and conditions. 

2. Secondly we used this blink detector to analyse all possible variations of some empirical measures in different subjects, during the baseline and the target period. In order to do this, we have developed a video interface embedded with some live statistical measures.

3. Finally, we have tried to analyse from a descriptive point of view the data displayed, with the intent to find out some analogies with the current research in the topic of lie detection and eye blinking.

Due to the type of experimental design, it was not feasible to compare our results with those obtained for example in Leal and Vrij: in our experiment, the subject’s activity was not the same during the baseline and the target period. Talking is a factor that definitly influences eye blinking. However, it was found a significant effect of the unexpected questions on the blink rate, especially when lying.

We believe that a more accurate experimental design and a possible live blink detection implementation could lead to a complete lie detector, which classifies a subject or a part
of his speech as a lie.
## References
- Anna Rita Bentivoglio, Susan B Bressman, Emanuele Cassetta, Donatella Carretta, Pietro Tonali, and Alberto Albanese. Analysis of blink rate patterns in normal subjects. Movement Disorders, 12(6):1028–1034, 1997.
- Sharon Leal and Aldert Vrij. Blinking during and after lying. Journal of Nonverbal Behavior, 32(4):187–194, 2008.
- Adrian Rosebrock. Eye blink detection with opencv, python, and dlib. URL https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/.
- Christos Sagonas, Georgios Tzimiropoulos, Stefanos Zafeiriou, and Maja Pantic. 300 faces in-the-wild challenge: The first facial landmark localization challenge. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 397–403, 2013.
- T Soukupova and Jan Cech. Real-time eye blink detection using facial landmarks. In 21st Computer Vision Winter Workshop (CVWW 2016), pages 1–8, 2016.
