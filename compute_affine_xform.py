# Author: Paola Donis Noriega

import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from numpy.linalg import pinv

def compute_affine_xform(matches,features1,features2,image1,image2):
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
        affine_xform (numpy.ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
    """
    
    affine_xform = np.zeros((3,3))
    threshold = 10.0 # how far away can the actual point be from the calculated point to still be considered an inlier
    
    # find combinations of matches to calculate the transformation matrix
    amount_matches = len(matches)
    comb = np.arange(amount_matches)
    amount_combinations = math.factorial(amount_matches) / (6 * math.factorial(amount_matches - 3))
    all_comb = np.zeros((amount_combinations, 3))
    ranks = np.zeros((1, amount_combinations))
    
    x = 0
    for p in combinations(comb, 3): # generate the possible combinations
        for q in range(0,len(p)):
            all_comb[x][q] = int(p[q])
        x +=1

    # generate transformation matrices that correspond to each combination
    for h in range(0, amount_combinations):
        transformation = find_matrix(h, matches, all_comb,features1, features2)

        rank = 0
        for k in range(0, amount_matches): # try each transformation matrix with all the points
            index1 = matches[k][0]
            index2 = matches[k][1]
            x = float(features1[index1][1])
            y = float(features1[index1][0])
            initial_point = np.matrix([[x],[y],[1.0]])
            x_prime = float(features2[index2][1])
            y_prime = float(features2[index2][0])
            calculated_point = np.dot(transformation, initial_point)
            calc_x = float(calculated_point[0,0])
            calc_y = float(calculated_point[1,0])
            if ((abs(calc_x - x_prime) < threshold) and (abs(calc_y - y_prime) < threshold)): # keep track of amount of inliers
                rank += 1
        ranks[0][h] = rank # store the amount of inliers per transformation matrix


    chosen_combination = np.argmax(ranks)
    h = chosen_combination
    affine_xform = find_matrix(h, matches, all_comb, features1, features2) # calculate the right transformation using the combination that resulted in more inliers

    return affine_xform

def find_matrix(h, matches, all_comb, features1, features2): # function that returns the transformation matrix
    index1 = matches[int(all_comb[h][0])]
    index2 = matches[int(all_comb[h][1])]
    index3 = matches[int(all_comb[h][2])]
    x1 = float(features1[index1[0]][1])
    y1 = float(features1[index1[0]][0])
    x2 = float(features1[index2[0]][1])
    y2 = float(features1[index2[0]][0])
    x3 = float(features1[index3[0]][1])
    y3 = float(features1[index3[0]][0])
    x1_prime = float(features2[index1[1]][1])
    y1_prime = float(features2[index1[1]][0])
    x2_prime = float(features2[index2[1]][1])
    y2_prime = float(features2[index2[1]][0])
    x3_prime = float(features2[index3[1]][1])
    y3_prime = float(features2[index3[1]][0])
    A = np.matrix([[x1, y1, 1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, x1, y1, 1.0],[x2, y2, 1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, x2, y2, 1],[x3, y3, 1.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, x3, y3, 1.0]])
    A_inv = pinv(A)
    b = [[x1_prime],[y1_prime],[x2_prime],[y2_prime],[x3_prime],[y3_prime]]
    t = np.dot(A_inv, b)
    transformation = np.matrix([[t[0,0], t[1,0], t[2,0]], [t[3,0], t[4,0], t[5,0]], [0.0, 0.0, 1.0]])

    return transformation
