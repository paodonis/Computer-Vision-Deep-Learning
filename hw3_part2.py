import cv2
import numpy as np
from skimage import feature
from skimage.feature import ORB


def feature_recognition():

    image1 = cv2.imread('hopkins1.JPG',0)
    image2 = cv2.imread('hopkinsw.JPG',0)
    (x,y) = np.shape(image1)
    print(x)
    print(y)
    #image1 = cv2.cvtColor(image1_color,cv2.COLOR_BGR2GRAY)
    #image2 = cv2.cvtColor(image2_color,cv2.COLOR_BGR2GRAY)
    image1 = np.asarray(image1)
    image2 = np.asarray(image2).reshape(-1)
    print(image1)

    detector_extractor1 = ORB(n_keypoints=5)
    detector_extractor2 = ORB(n_keypoints=5)
    detector_extractor1.detect_and_extract(image1)
    detector_extractor2.detect_and_extract(image2)

    print(detector_extractor1)

feature_recognition()
