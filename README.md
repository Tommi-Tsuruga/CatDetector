# CatDetector
This project uses two types of face detection for detecting cats in images for use in assessing the appropriate detection technique to use.  The Haar seems to be slightly faster and requires no training time, as openCV has Haar Cascades built in for cats. From preliminary testing,the HoG and SVM model is more accurate.  

Haar Detector:
The Haar model is built using openCV's built in cascade classifier. This classifier slides kernels across an image comparing the features to the built in model, if enough are similar, it will output that the image contains one or more cats. Potential issues include multiple bounding boxes being predicted for a single cat. It might be worthwhile to play with the non-max surpression thresholds.

![Haar-featurs](https://docs.opencv.org/2.4/_images/haarfeatures.png)

Example input image and output image with bounding box drawn on

![Cat-image](https://i.imgur.com/5hKOYSD.png)
![Haar-Cat](https://i.imgur.com/vSToD1s.png)

HoG Features with SVM classifier:
Uses openCV to extract HoG features before training an SVM classifier for detection. Hog features contain information about the gradients. Following this feature extraction, these gradients are loaded into a Support Vector Machine to train on classification. Model building takes a long time but once the model is built extracting the features is relatively quick.
Hog features look like:

![alt text](https://i.imgur.com/2cqHcoc.png)

## Getting Started
### Prerequisites
Both classifiers require python3
Currently this project requires a local copy of the dataset found here for the SVM to build properly. [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) The /images folder should be located in the same directory as the cat_hog_train.py file

### Installing
Using a virtual enviroment is recommended. From there the requirements.txt can be installed. An example installation on linux might look like:

```
python3 -m virtualenv env
source env/bin/activate
pip install -r /path/to/requirements.txt
```

From there running
```
python3 cat_haar.py
```
or
```
python3 cat_hog_train.py
```
should work.

The requirements.txt should install the appropriate dependencies for both classifiers. 
