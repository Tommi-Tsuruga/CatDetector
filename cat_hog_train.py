from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import random
import numpy as np
import cv2 as cv
import matplotlib
from PIL import Image
from skimage import feature


print("Loading images into memory")
cat_imgs, noncat_imgs = [], []
trainfile = "annotations/trainval.txt"
file = open(trainfile, "r")
for line in file:
    entry = line.split()
    if entry[2] == "1":
        imagePath = 'images/' + entry[0] + '.jpg'
        print(imagePath)
        image = cv.imread(imagePath)
        graycats = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        cat_resized = resize(graycats, (250, 250))
        cat_imgs.append(cat_resized)
        continue

    else:
        imagePath = 'images/' + entry[0] + '.jpg'
        image = cv.imread(imagePath)
        graynotcats = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        notcat_resized = resize(graynotcats, (250, 250))
        noncat_imgs.append(notcat_resized)
        continue

cat_imgs, noncat_imgs = np.asarray(cat_imgs), np.asarray(noncat_imgs)
total_cats, total_noncats = cat_imgs.shape[0], noncat_imgs.shape[0]

print("... Done")
print("Cat images shape: ", cat_imgs.shape)
print("Non-cat images shape: ", noncat_imgs.shape)

print("Extracting features")

cat_features, noncat_features = [], []

for img in cat_imgs:
    HOG = feature.hog(img, orientations=11, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    cat_features.append(HOG)

for img in noncat_imgs:
    noncatHOG = feature.hog(img, orientations=11, pixels_per_cell=(16, 16), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    noncat_features.append(noncatHOG)

cat_features = np.asarray(cat_features)
noncat_features = np.asarray(noncat_features)
print("Extracting done")
print("Cat images shape: ", cat_imgs.shape)
print("Non-cat images shape: ", noncat_imgs.shape)

print("Scaling features...")

unscaled_x = np.vstack((cat_features, noncat_features)).astype(np.float64)
scaler = StandardScaler().fit(unscaled_x)
x = scaler.transform(unscaled_x)
y = np.hstack((np.ones(total_cats), np.zeros(total_noncats)))

print("Scaling Done")
print("x shape: ", x.shape, "y shape: ", y.shape)

print("Training classifier")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = random.randint(1, 100))
svc = LinearSVC()
svc.fit(x_train, y_train)
accuracy = svc.score(x_test, y_test)

print("Training Done")
print("Accuracy: ", np.round(accuracy, 4))
