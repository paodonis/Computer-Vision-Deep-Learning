'''
Program that calculates the fundamental matrix and draws the epipolar lines.
It uses ORB to find the features, match_features.py to find the matches and
compute_proj_xform to choose the right transformations and find the inliers
using RANSAC.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from match_features import match_features
from compute_proj_xform import compute_proj_xform
import random


def detect_features(image): # function that returns a tuple with the x and y pixels where a features is located
    orb = cv2.ORB_create()
    kp = orb.detect(image,None)
    all_x = list()
    all_y = list()

    for keyPoint in kp:
        x = int(keyPoint.pt[0])
        y = int(keyPoint.pt[1])
        all_x.append(x)
        all_y.append(y)
    coords = []
    for i in range(0, 400): # use 400 of the features it returns
        coords.append((all_y[i], all_x[i]))
        tup = tuple(coords)
        
    return tup
   
image1 = cv2.imread('hopkins1.JPG',0) # open the images
image1_color = cv2.imread('hopkins1.JPG')
image_matches_1 = image1_color.copy()
tup_1 = detect_features(image1)

image2 = cv2.imread('hopkins2.JPG',0)
image2_color = cv2.imread('hopkins2.JPG')
image_matches_2 = image2_color.copy()
tup_2 = detect_features(image2)

(vertical_1, horizontal_1) = np.shape(image1)
(vertical_2, horizontal_2) = np.shape(image2)


matches = match_features(tup_1, tup_2, image1, image2) # match_features.py returns a tuple with the index of the pairs


transformation = compute_proj_xform(matches, tup_1, tup_2, image1, image2) # returns the transformation that results in more inliers

threshold = 20.0
two_images_matches = np.hstack((image_matches_1, image_matches_2))
correct_matches = []

for k in range(0, len(matches)): # using the projective transformation matrix, check if each match is an inlier (green) or outlier (blue)
    index1 = matches[k][0]
    index2 = matches[k][1]
    x = float(tup_1[index1][1])
    y = float(tup_1[index1][0])
    initial_point = np.matrix([[x],[y],[1.0]])
    x_prime = float(tup_2[index2][1])
    y_prime = float(tup_2[index2][0])
    calc_x = (transformation[0,0] * x + transformation[0,1] * y + transformation[0,2])/(transformation[2,0] * x + transformation[2,1] * y + transformation[2,2])
    calc_y = (transformation[1,0] * x + transformation[1,1] * y + transformation[1,2])/(transformation[2,0] * x + transformation[2,1] * y + transformation[2,2])
    if ((abs(calc_x - x_prime) < threshold) and (abs(calc_y - y_prime) < threshold)): # check if the match is described by the transformation
        correct_matches.append([x,y, x_prime, y_prime])
        cv2.circle(two_images_matches, (int(x), int(y)), 4, (255, 255, 0), 3) # graph features and matching lines for the inliers
        cv2.circle(two_images_matches, (int(x_prime) + horizontal_1, int(y_prime)), 4, (255, 255, 0), 2)
        cv2.line(two_images_matches,(int(x), int(y)),(int(x_prime) + horizontal_1, int(y_prime)),(255, 255, 0),2)

selected_points = [0, 1, 2, 8, 11, 12, 13, 14] # since many features are points nearby, selected points are points that are significantly far apart from each other

c = correct_matches

M = np.zeros((8, 9))

# generate the M matrix to calculate the fundamental matrix
for i in range(0, len(selected_points)):
    x = selected_points[i]
    for y in range(0,9):
        if (y == 0):
            M[i][y] = c[x][0] * c[x][2]
        elif (y == 1):
            M[i][y] = c[x][0] * c[x][3]
        elif (y == 2):
            M[i][y] = c[x][0]
        elif (y == 3):
            M[i][y] = c[x][1] * c[x][2]
        elif (y == 4):
            M[i][y] = c[x][1] * c[x][3]
        elif (y == 5):
            M[i][y] = c[x][1]
        elif (y == 6):
            M[i][y] = c[x][2]
        elif (y == 7):
            M[i][y] = c[x][3]
        elif (y == 8):
            M[i][y] = 1
        
U, s, V = np.linalg.svd(np.matrix(M), full_matrices=False)

chosen_V = V[-1] # choose the V with the smallest singular value


(sizey, sizex) = np.shape(chosen_V)

for l in range(0, sizex): # normalize V so that the last value is a 1
        chosen_V[0,l] = chosen_V[0,l]/chosen_V[0,8]

fundamental_matrix = np.reshape(chosen_V, (3, 3)) # reshape the fundamental matrix into a 3x3 matrix
fundamental_matrix = np.matrix.transpose(fundamental_matrix) # shape the matrix the right way since V = [f11, f21, f31, f12, f22, f32, f31, f32, f33]^T


U,s,V = np.linalg.svd(fundamental_matrix, full_matrices = False) # find the singular values to set the smallest one to 0 so that the matrix is rank 2

D_prime = np.diag(s) # matrix with the singular values in its diagonal
D_prime[2][2] = 0 #set the smallest singular value to 0

F = np.dot(U, np.dot(D_prime, V)) # find the final fundamental matrix


selected_points = [0, 1, 2, 4, 8, 11, 12, 13, 14] # chosen points to graph

for j in range(0, len(selected_points)):
    i = selected_points[j]
    # find the features that will be used for the calculations of the epipolar line
    u = c[i][0]
    v = c[i][1]
    u_point2 = c[i][2]
    v_point2 = c[i][3]

    # calculate the factors a,b,c from the fundamental matrix
    a = F[0,0] * u + F[0,1] * v + F[0,2]
    b = F[1,0] * u + F[1,1] * v + F[1,2]
    c_factor = F[2,0] * u + F[2,1] * v + F[2,2]

    # calculate the two endpoints of the line at the edges of the image
    u1 = 1
    v1 = int((-c_factor - a* u1)/b)
    if (v1 > (vertical_2-1) or v1 < 1):
        v1 = 1
        u1 = int((-c_factor-b*v1)/a)

    u2 = horizontal_2 - 1
    v2 = int((-c_factor - a* u2)/b)
    if (v2 > (vertical_2 - 1) or v2 < 1):
        v2 = (vertical_2 - 1)
        u2 = int((-c_factor-b*v2)/a)

    # generate random colors for each point
    red = random.randrange(0,255,1)
    green = random.randrange(0,255,1)
    blue = random.randrange(0,255,1)

    cv2.line(image2_color,(u1,v1),(u2,v2),(red, green, blue),2) # graph the epipolar lines
    cv2.circle(image1_color, (int(u),int(v)), 4, (red, green, blue), 2) # graph the points in image1
    cv2.circle(image2_color, (int(u_point2),int(v_point2)), 4, (red, green, blue), 2) # graph the points in image2


combined_image = np.hstack((image1_color, image2_color)) # combine both images to be displayed next to each other

cv2.imshow('matches', two_images_matches) # show the image of the matches generated and of the epipolar lines
cv2.imshow('epipolar lines', combined_image)
cv2.waitKey()
