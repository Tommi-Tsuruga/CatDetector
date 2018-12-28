import argparse
import cv2
 
 
#usage:
#requires dataset folder in same folder
#use with python2.7 only
# python2.7 cat_haar.py --imagefolder annotations/test.txt
#parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagefolder", required=True,
	help="path to the test folder")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())


testfile = "annotations/test.txt"
file = open(testfile, "r")
totalcats = 0
detectedcats = 0
#need to convert text file lines to paths to images
for line in file:
    entry = line.split()
    if entry[2] == "1":
        totalcats += 1
        imagePath = 'images/' + entry[0] + '.jpg'
        #print(imagePath, "first")
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cat = cv2.resize(gray, (400, 400))
        detector = cv2.CascadeClassifier(args["cascade"])
        #modified cat
        cats = detector.detectMultiScale(cat, scaleFactor=1.15,
        minNeighbors=10, minSize=(75, 75))
        for (i, (x, y, w, h)) in enumerate(cats):
            detectedcats += 1

        continue
print "Total cats:", totalcats
print "Detected Cats:", detectedcats
average = float(float(detectedcats)/float(totalcats))
print 'Average:', average*100
