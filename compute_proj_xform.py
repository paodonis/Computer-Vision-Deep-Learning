# Author: Paola Donis Noriega
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from numpy.linalg import inv

def compute_proj_xform(matches,features1,features2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        proj_xform (numpy.ndarray): a 3x3 Projective transformation matrix between the two images, computed using the matches.
    """   
    proj_xform = np.zeros((3,3))

    threshold = 10.0
    
    ##################### FIND ALL COMBINATIONS OF 3 MATCHES ####################################
    amount_matches = len(matches)
    comb = np.arange(amount_matches)
    amount_combinations = math.factorial(amount_matches) / (24 * math.factorial(amount_matches - 4))
    all_comb = np.zeros((amount_combinations, 4))
    ranks = np.zeros((1, amount_combinations))
    
    x = 0
    for p in combinations(comb, 4):
        for q in range(0,len(p)):
            all_comb[x][q] = int(p[q])
        x +=1

    ################## CREATE EACH TRANSFORMATION MATRIX #########################################

    for h in range(0, amount_combinations):
        index1 = matches[int(all_comb[h][0])]
        index2 = matches[int(all_comb[h][1])]
        index3 = matches[int(all_comb[h][2])]
        index4 = matches[int(all_comb[h][3])]
        x1 = float(features1[index1[0]][1])
        y1 = float(features1[index1[0]][0])
        x2 = float(features1[index2[0]][1])
        y2 = float(features1[index2[0]][0])
        x3 = float(features1[index3[0]][1])
        y3 = float(features1[index3[0]][0])
        x4 = float(features1[index4[0]][1])
        y4 = float(features1[index4[0]][0])
        x1_prime = float(features2[index1[1]][1])
        y1_prime = float(features2[index1[1]][0])
        x2_prime = float(features2[index2[1]][1])
        y2_prime = float(features2[index2[1]][0])
        x3_prime = float(features2[index3[1]][1])
        y3_prime = float(features2[index3[1]][0])
        x4_prime = float(features2[index4[1]][1])
        y4_prime = float(features2[index4[1]][0])

        A = np.matrix([[x1,y1, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x1_prime) * x1), (-1.0 * x1_prime * y1), (-1.0 * x1_prime)],[0.0, 0.0, 0.0, x1, y1, 1.0, (-1.0 * y1_prime * x1), (-1.0 * y1_prime * y1), (-1.0* y1_prime)],[x2,y2, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x2_prime) * x2), (-1.0 * x2_prime * y2), (-1.0 * x2_prime)],[0.0, 0.0, 0.0, x2, y2, 1.0, (-1.0 * y2_prime * x2), (-1.0 * y2_prime * y2), (-1.0* y2_prime)],[x3,y3, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x3_prime) * x3), (-1.0 * x3_prime * y3), (-1.0 * x3_prime)],[0.0, 0.0, 0.0, x3, y3, 1.0, (-1.0 * y3_prime * x3), (-1.0 * y3_prime * y3), (-1.0* y3_prime)],[x4,y4, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x4_prime) * x4), (-1.0 * x4_prime * y4), (-1.0 * x4_prime)],[0.0, 0.0, 0.0, x4, y4, 1.0, (-1.0 * y4_prime * x4), (-1.0 * y4_prime * y4), (-1.0* y4_prime)]])
                
        U, s, V = np.linalg.svd(A, full_matrices=True)
        minimum_eigenvector = V[-1]

        (sizey, sizex) = np.shape(minimum_eigenvector)

        for l in range(0, sizex):
            minimum_eigenvector[0,l] = minimum_eigenvector[0,l]/minimum_eigenvector[0,8]
        
        transformation = np.reshape(minimum_eigenvector, (3, 3)) # form the transformation matrix using the minimum vector
        
        rank = 0
        
        for k in range(0, amount_matches):
            index1 = matches[k][0]
            index2 = matches[k][1]
            x = float(features1[index1][1])
            y = float(features1[index1][0])
            initial_point = np.matrix([[x],[y],[1.0]])
            x_prime = float(features2[index2][1])
            y_prime = float(features2[index2][0])
            if ((transformation[2,0] * x2 + transformation[2,1] * y2 + transformation[2,2]) == 0.0):
                calc_x = 0.0
                calc_y = 0.0
            else:
                if (transformation[2,0] * x + transformation[2,1] * y + transformation[2,2] == 0.0):
                    calc_x = transformation[0,0] * x + transformation[0,1] * y + transformation[0,2]
                    calc_y = transformation[1,0] * x + transformation[1,1] * y + transformation[1,2]
                else:
                    calc_x = (transformation[0,0] * x + transformation[0,1] * y + transformation[0,2])/(transformation[2,0] * x + transformation[2,1] * y + transformation[2,2])
                    calc_y = (transformation[1,0] * x + transformation[1,1] * y + transformation[1,2])/(transformation[2,0] * x + transformation[2,1] * y + transformation[2,2])

            if ((abs(calc_x - x_prime) < threshold) and (abs(calc_y - y_prime) < threshold)): # check if it would be considered an inlier or outlier
                rank += 1
        ranks[0][h] = rank


    chosen_combination = np.argmax(ranks) # find the transformation with more inliers
    h = chosen_combination
    index1 = matches[int(all_comb[h][0])]
    index2 = matches[int(all_comb[h][1])]
    index3 = matches[int(all_comb[h][2])]
    index4 = matches[int(all_comb[h][3])]
    
    x1 = float(features1[index1[0]][1])
    y1 = float(features1[index1[0]][0])
    x2 = float(features1[index2[0]][1])
    y2 = float(features1[index2[0]][0])
    x3 = float(features1[index3[0]][1])
    y3 = float(features1[index3[0]][0])
    x4 = float(features1[index4[0]][1])
    y4 = float(features1[index4[0]][0])
    x1_prime = float(features2[index1[1]][1])
    y1_prime = float(features2[index1[1]][0])
    x2_prime = float(features2[index2[1]][1])
    y2_prime = float(features2[index2[1]][0])
    x3_prime = float(features2[index3[1]][1])
    y3_prime = float(features2[index3[1]][0])
    x4_prime = float(features2[index4[1]][1])
    y4_prime = float(features2[index4[1]][0])
    
    # generate the transformation matrix with best results again
    A = np.matrix([[x1,y1, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x1_prime) * x1), (-1.0 * x1_prime * y1), (-1.0 * x1_prime)],[0.0, 0.0, 0.0, x1, y1, 1.0, (-1.0 * y1_prime * x1), (-1.0 * y1_prime * y1), (-1.0* y1_prime)],[x2,y2, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x2_prime) * x2), (-1.0 * x2_prime * y2), (-1.0 * x2_prime)],[0.0, 0.0, 0.0, x2, y2, 1.0, (-1.0 * y2_prime * x2), (-1.0 * y2_prime * y2), (-1.0* y2_prime)],[x3,y3, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x3_prime) * x3), (-1.0 * x3_prime * y3), (-1.0 * x3_prime)],[0.0, 0.0, 0.0, x3, y3, 1.0, (-1.0 * y3_prime * x3), (-1.0 * y3_prime * y3), (-1.0* y3_prime)],[x4,y4, 1.0, 0.0, 0.0, 0.0, (-1.0 *(x4_prime) * x4), (-1.0 * x4_prime * y4), (-1.0 * x4_prime)],[0.0, 0.0, 0.0, x4, y4, 1.0, (-1.0 * y4_prime * x4), (-1.0 * y4_prime * y4), (-1.0* y4_prime)]])


    U, s, V = np.linalg.svd(A, full_matrices=True)
    minimum_eigenvector = V[-1]
    

    (sizey, sizex) = np.shape(minimum_eigenvector)

    for l in range(0, sizex):
            minimum_eigenvector[0,l] = minimum_eigenvector[0,l]/minimum_eigenvector[0,8]
        
    proj_xform = np.reshape(minimum_eigenvector, (3, 3))


    return proj_xform
