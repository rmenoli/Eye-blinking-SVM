# Analysis of Lying through an Eye Blink Detection Algorithm
Lie detection using physiological methods has been utilized for the past 75 years, starting with the work of Benussi, with varying rates of success in identification of the guilty.
More recently, eye blink measures have attracted considerable attention, since they have been related to cognitive processes. They can be recorded easily without the subject’s awareness and without the application of electrodes.

The first part of our project consisted of acquiring the background knowledge in the domain of cognitive and behavioral sciences with a focus on the relevant researches on eye blink and lie detection.

The second part was more technical and regarded the research of the free tools already available for eye blink detection with a good performance in terms of accuracy. To our knowledge, the only freely-available software for blink detection is built on the OpenCV Python modules and is based on a metric called eye aspect ratio (EAR). Despite the acceptable results in live blink detection, this ready-solution performed poorly in different video conditions.

Therefore, the third part of the project consisted of building, testing and validating a Support Vector Machine (SVM) classifier using the EAR features.

Finally, we created an interface for the video to analyse from
a lie detection point of view, together with some useful descriptive plots of blinking measures.

## OpenCV and Facial Landmarks
The implementation which follows is based on the tutorial from the blog Pyimagesearch (see Rosebrock) which, in turn, takes as its starting point the work of Soukupova and Cech for the the metric called eye aspect ratio (EAR). The 68 facial landmarks are based on the iBUG 300-W dataset (see Sagonas et al. and figure below), which the dlib facial landmark predictor was trained on.
![](https://github.com/rmenoli/Eye-blinking-SVM/blob/master/images/facial_landmarks_68markup.jpg | width=100)

## References
- Adrian Rosebrock. Eye blink detection with opencv, python, and dlib. URL https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/.
- Christos Sagonas, Georgios Tzimiropoulos, Stefanos Zafeiriou, and Maja Pantic. 300 faces in-the-wild challenge: The first facial landmark localization challenge. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 397–403, 2013.
- T Soukupova and Jan Cech. Real-time eye blink detection using facial landmarks. In 21st Computer Vision Winter Workshop (CVWW 2016), pages 1–8, 2016.