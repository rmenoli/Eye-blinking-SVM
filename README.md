# Analysis of Lying through an Eye Blink Detection Algorithm
Lie detection using physiological methods has been utilized for the past 75 years, starting with the work of Benussi, with varying rates of success in identification of the guilty.
More recently, eye blink measures have attracted considerable attention, since they have been related to cognitive processes. They can be recorded easily without the subject’s awareness and without the application of electrodes.

The first part of our project consisted of acquiring the background knowledge in the domain of cognitive and behavioral sciences with a focus on the relevant researches on eye blink and lie detection.

The second part was more technical and regarded the research of the free tools already available for eye blink detection with a good performance in terms of accuracy. To our knowledge, the only freely-available software for blink detection is built on the OpenCV Python modules and is based on a metric called eye aspect ratio (EAR). Despite the acceptable results in live blink detection, this ready-solution performed poorly in different video conditions.

Therefore, the third part of the project consisted of building, testing and validating a Support Vector Machine (SVM) classifier using the EAR features.

Finally, we created an interface for the video to analyse from
a lie detection point of view, together with some useful descriptive plots of blinking measures.

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

### Validation
The SVM blink detector was firstly validated on some videos from the iBUG 300-W dataset and its performance was therefore compared with the OpenCV blink detector (see Rosebrock). The aim of this stage was to test the two system’s blink detection ability, independently of variations in pose, expression, illumination, background, occlusion, and image quality. 

Following what has been done in Sagonas et al., the following three scenarios have been considered:
- Scenario 1 (Top): A number of testing videos will be of people recorded in well-lit conditions displaying arbitrary expressions in various head poses. This scenario aims to evaluate algorithms that could be suitable for facial motion analysis in laboratory and naturalistic well-lit conditions.
- Scenario 2 (Middle): A number of testing videos will be of people recorded in unconstrained conditions, displaying arbitrary expressions in various head poses but without large occlusions. This scenario aims to evaluate algorithms that could be suitable for facial motion analysis in real-world human-computer interaction applications.
- Scenario 3 (Bottom): A number of testing videos will be of people recorded in completely unconstrained conditions including the illumination conditions, occlusions, makeup, expression, head pose, etc. This scenario aims to assess the performance of facial landmark tracking in arbitrary conditions.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/Validation_Scenarios.png" width="350"/>
</p>

Secondly, the validation procedure was done also for videos of friends, relatives and acquaintances; the results for two videos are summarized below. We can notice that also for new and longer videos (the lengths are 3:45 and 7:30 minutes, respectively for Bianca at the top and Filippo at the bottom) the validation procedure shows a significant improvement in the SVM blink detector in terms of both precision and recall, especially when occlusions occur (i.e. Filippo wears glasses). Moreover, the overall SVM classifier performance is better in Filippo, due to the frame rate the video was recorded at: Bianca was recorded at 14 fps, while Filippo at 29 fps. The SVM blink detector was trained with 30 fps videos, so this difference in performances was expected, but unavoidable.

<p align="center"> 
<img src="https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/Validation_Experiments.png" width="350"/>
</p>

## References
- Adrian Rosebrock. Eye blink detection with opencv, python, and dlib. URL https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/.
- Christos Sagonas, Georgios Tzimiropoulos, Stefanos Zafeiriou, and Maja Pantic. 300 faces in-the-wild challenge: The first facial landmark localization challenge. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 397–403, 2013.
- T Soukupova and Jan Cech. Real-time eye blink detection using facial landmarks. In 21st Computer Vision Winter Workshop (CVWW 2016), pages 1–8, 2016.