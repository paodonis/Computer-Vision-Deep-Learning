'''
Program that calculates the fundamental matrix and draws the epipolar lines.
It uses ORB descriptors and finds the matches using euclidean distance.
Then the fundamental matrix is calculated using groups of 8 matches and RANSAC
is used to classify inliers.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
import math
from itertools import combinations


def detect_features(image): # function that returns keypoints and descriptors
    orb = cv2.ORB_create()

    kp = orb.detect(image, None)
    kp, des = orb.compute(image, kp)

    return kp, des
    
image1 = cv2.imread('hopkins1.JPG',0) # open the images
image1_color = cv2.imread('hopkins1.JPG')
image_matches_1 = image1_color.copy()
kp1, descriptors_image1 = detect_features(image1)
all_x = list()
all_y = list()
    
for keyPoint in kp1: # create a tuple with the key points
    x = int(keyPoint.pt[0])
    y = int(keyPoint.pt[1])
    all_x.append(x)
    all_y.append(y)
coords = []
for i in range(0, len(kp1)):
    coords.append((all_y[i], all_x[i]))
    tup_1 = tuple(coords)


image2 = cv2.imread('hopkins2.JPG',0) # open image 2 and create copies to be displayed later
image2_color = cv2.imread('hopkins2.JPG')
image_matches_2 = image2_color.copy()
kp2, descriptors_image2 = detect_features(image2)
all_x = list()
all_y = list()

for keyPoint in kp2: # create a tuple with the keypoints
    x = int(keyPoint.pt[0])
    y = int(keyPoint.pt[1])
    all_x.append(x)
    all_y.append(y)
coords = []
for i in range(0, len(kp2)):
    coords.append((all_y[i], all_x[i]))
    tup_2 = tuple(coords)


(vertical_1, horizontal_1) = np.shape(image1)
(vertical_2, horizontal_2) = np.shape(image2)

chosen_matches = []
thresh = 0.6 # threshold that controls the amount of matches

for index1, desc1 in enumerate(descriptors_image1): # calculate the eucledian distance of the descriptors to find the matches
    distances = []
    for index2, desc2 in enumerate(descriptors_image2):
        euclidean_distance = (desc1 ^ desc2).sum()
        distances.append(euclidean_distance)
        
    min_distance = min(distances)
    next_min_dist = float(np.partition(distances, 1)[1])
    
    if (thresh * next_min_dist > min_distance):
        index_neighbor = distances.index(min_distance)
        chosen_matches.append([index1, index_neighbor])

matches = tuple(chosen_matches)

amount_matches = len(matches)

for m in range(0, amount_matches): # graph the matches
        tup_image_1 = tup_1[m]
        tup_image_2 = tup_2[m]
        cv2.circle(image_matches_1, (tup_1[matches[m][0]][1], tup_1[matches[m][0]][0]), 1, (0, 255, 0), 3)
        cv2.circle(image_matches_2, (tup_2[matches[m][1]][1], tup_2[matches[m][1]][0]), 1, (0, 255, 0), 3)

two_images_match = np.hstack((image_matches_1, image_matches_2))

for m in range(0, amount_matches):
    cv2.line(two_images_match,(tup_1[matches[m][0]][1], tup_1[matches[m][0]][0]),(tup_2[matches[m][1]][1] + horizontal_1, tup_2[matches[m][1]][0]),(255,0,0),2)


##################### FIND ALL COMBINATIONS OF 8 MATCHES ####################################
    
amount_matches = len(matches)
comb = np.arange(amount_matches)

amount_combinations = math.factorial(amount_matches) / (math.factorial(8) * math.factorial(amount_matches - 8))
all_comb = np.zeros((amount_combinations, 8))
ranks = np.zeros((1, amount_combinations))

x = 0
for p in combinations(comb, 8):
    for q in range(0,len(p)):
        all_comb[x][q] = int(p[q])
    x +=1

c = [] # matrix with all the matches in the format [x1, y1, x2, y2]

for k in range(0, len(matches)):
    index1 = matches[k][0]
    index2 = matches[k][1]
    x1 = float(tup_1[index1][1])
    y1 = float(tup_1[index1][0])
    x2 = float(tup_2[index2][1])
    y2 = float(tup_2[index2][0])
    c.append([x1, y1, x2, y2])

def generate_M(selected_points): # function that generates the fundamental matrix
    M = np.zeros((8, 9))

    # generate the M matrix to calculate the fundamental matrix
    for i in range(0, len(selected_points)):
        x = int(selected_points[i])
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
            
    U, s, V = np.linalg.svd(np.matrix(M))

    chosen_V = V[-1] # choose the V with the smallest singular value


    (sizey, sizex) = np.shape(chosen_V)
    
    for l in range(0, sizex): # normalize V so that the last value is a 1
            chosen_V[0,l] = chosen_V[0,l]/chosen_V[0,8]
    
    fundamental_matrix = np.reshape(chosen_V, (3, 3)) # reshape the fundamental matrix into a 3x3 matrix
    fundamental_matrix = np.matrix.transpose(fundamental_matrix) # shape the matrix the right way since V = [f11, f21, f31, f12, f22, f32, f31, f32, f33]^T


    U,s,V = np.linalg.svd(fundamental_matrix) # find the singular values to set the smallest one to 0 so that the matrix is rank 2

    D_prime = np.diag(s) # matrix with the singular values in its diagonal
    D_prime[2][2] = 0 #set the smallest singular value to 0

    F = np.dot(U, np.dot(D_prime, V)) # find the final fundamental matrix
    F= F/F[2,2]
    
    return F

thresh_mat = 0.03 # threshold that determines which matches agree with each calculated fundamental matrix

for h in range(0, amount_combinations): # iterate through all combinations
    h = h
    selected_points = np.zeros((8,1))
    for p in range(0,8):
        selected_points[p] = all_comb[h][p]
    F = generate_M(selected_points)
    rank = 0
    
    for k in range(0, amount_matches):
        index1 = matches[k][0]
        index2 = matches[k][1]
        x = float(tup_1[index1][1])
        y = float(tup_1[index1][0])
        left = np.matrix([x,y,1.0])
        x_prime = float(tup_2[index2][1])
        y_prime = float(tup_2[index2][0])
        right = np.matrix([[x_prime],[y_prime],[1.0]])

        value = np.dot(np.dot(left, F), right)
        if (value < thresh_mat): # check if each match is described by the calculated fundamental matrix
            rank += 1
    ranks[0][h] = rank # store the rankings to see which one agrees with more points according to RANSAC


h = np.argmax(ranks) # chosen combination of matches

selected_points = np.zeros((8,1))
for p in range(0,8):
    selected_points[p] = all_comb[h][p]
F = generate_M(selected_points) # generate the matrix that will be used


for j in range(0, len(selected_points)):
    i = int(selected_points[j])

    u = c[i][0]
    v = c[i][1]

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
    cv2.circle(image1_color, (int(u),int(v)), 4, (red, green, blue), 2) # graph the points in image1 to which the epipolar lines correspond to

combined_image = np.hstack((image1_color, image2_color)) # combine both images to be displayed next to each other

cv2.imshow('matches', two_images_match) # show the image of the matches generated and of the epipolar lines
cv2.imshow('epipolar lines', combined_image)
cv2.waitKey()
