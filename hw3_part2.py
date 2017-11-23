import numpy as np
import cv2
from matplotlib import pyplot as plt
from match_features import match_features
from compute_proj_xform import compute_proj_xform
from compute_affine_xform import compute_affine_xform
import random



def detect_features(image):
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
    for i in range(0, 400):
        coords.append((all_y[i], all_x[i]))
        tup = tuple(coords)
        
    return tup
   
image1 = cv2.imread('hopkins1.JPG',0)
image1_color = cv2.imread('hopkins1.JPG')
image_matches_1 = image1_color.copy()
tup_1 = detect_features(image1)

image2 = cv2.imread('hopkins2.JPG',0)
image2_color = cv2.imread('hopkins2.JPG')
image_matches_2 = image2_color.copy()
tup_2 = detect_features(image2)

(vertical_1, horizontal_1) = np.shape(image1)
(vertical_2, horizontal_2) = np.shape(image2)


matches = match_features(tup_1, tup_2, image1, image2)


transformation = compute_proj_xform(matches, tup_1, tup_2, image1, image2)

threshold = 20.0
border = 0
inliers = []
two_images_matches = np.hstack((image_matches_1, image_matches_2))

print("matches", len(matches))

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
    if ((abs(calc_x - x_prime) < threshold) and (abs(calc_y - y_prime) < threshold)):
        correct_matches.append([x,y, x_prime, y_prime])
        cv2.circle(two_images_matches, (int(x), int(y)), 4, (255, 255, 0), 3) # graph features and lines using green for inliers
        cv2.circle(two_images_matches, (int(x_prime) + horizontal_1, int(y_prime)), 4, (255, 255, 0), 2)
        cv2.line(two_images_matches,(int(x), int(y)),(int(x_prime) + horizontal_1, int(y_prime)),(255, 255, 0),2)

print(correct_matches)
c = correct_matches[0:8]

selected_points = [0, 1, 2, 8, 11, 12, 13, 14]

print("matches used", c)

c = correct_matches ####

M = np.zeros((8, 9))

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
        
print("M", M)

U, s, V = np.linalg.svd(np.matrix(M), full_matrices=False)

print("first U", U)
print("first s", s)
print("first V", V)

chosen_V = V[-1]

print("chosen V", chosen_V)


(sizey, sizex) = np.shape(chosen_V)


for l in range(0, sizex):
        chosen_V[0,l] = chosen_V[0,l]/chosen_V[0,8]

        
print("normalized V", chosen_V)

fundamental_matrix = np.reshape(chosen_V, (3, 3))
fundamental_matrix = np.matrix.transpose(fundamental_matrix)

print("fundamental matrix", fundamental_matrix)


U,s,V = np.linalg.svd(fundamental_matrix, full_matrices = False)

print("second U", U)
print("second s", s)
print("second V", V)

D_prime = np.diag(s)
D_prime[2][2] = 0

print("D_prime", D_prime)

F = np.dot(U, np.dot(D_prime, V))

print('F', F)


selected_points = [0, 1, 2, 4, 8, 11, 12, 13, 14]

for j in range(0, len(selected_points)):
    i = selected_points[j]

    u = c[i][0]
    v = c[i][1]
    print('u', u, 'v', v)
    u_point2 = c[i][2]
    v_point2 = c[i][3]


    a = F[0,0] * u + F[0,1] * v + F[0,2]
    b = F[1,0] * u + F[1,1] * v + F[1,2]
    c_factor = F[2,0] * u + F[2,1] * v + F[2,2]


    print('a', a, 'b', b, 'c', c_factor)


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

    print('point1', u1,v1, 'point2', u2, v2)

    red = random.randrange(0,255,1)
    green = random.randrange(0,255,1)
    blue = random.randrange(0,255,1)

    cv2.line(image2_color,(u1,v1),(u2,v2),(red, green, blue),2)
    cv2.circle(image1_color, (int(u),int(v)), 4, (red, green, blue), 2)
    cv2.circle(image2_color, (int(u_point2),int(v_point2)), 4, (red, green, blue), 2)


combined_image = np.hstack((image1_color, image2_color))


cv2.imshow('matches', two_images_matches)
cv2.imshow('epipolar lines', combined_image)
cv2.waitKey()
