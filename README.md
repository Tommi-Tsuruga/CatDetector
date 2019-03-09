# CatDetector
This project uses two types of face detection for detecting cats in images. 

Haar Detector:
Uses a sliding window cascade classifier with trained haar features to detect cat faces in images
![Haar-featurs](https://docs.opencv.org/2.4/_images/haarfeatures.png)


HoG Features with SVM classifier:
Uses openCV to extract HoG features before training an SVM classifier for detection
![alt text](https://i.imgur.com/2cqHcoc.png)

## Getting Started
The requirements.txt should install the appropriate dependencies for both classifiers
