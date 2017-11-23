import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def match_features(feature_coords1,feature_coords2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """

    (vertical, horizontal) = np.shape(feature_coords1) # find the dimensions of the tuple containing the features
    amount_points_1 = len(feature_coords1) # calculate number of features per image
    amount_points_2 = len(feature_coords2)
    rank_1 = np.zeros((amount_points_1,amount_points_2), dtype = "float32") # matrices that will store values achieved by each combination
    rank_2 = np.zeros((amount_points_2,amount_points_1), dtype = "float32")
    (height, width) = np.shape(image1)


    (image1, mean_value1) = mean_image(image1) # calculate the mean of the pixels and subtract it from every pixel in a function defined below 
    (image2, mean_value2) = mean_image(image2)

    image1 = variance_image(image1, mean_value1) # calculate the variance of the pixels and divide each pixel by the variance in order to normalize the image
    image2 = variance_image(image2, mean_value2)


    for p in range(0, amount_points_1):
        y_position_1 = feature_coords1[p][0]
        x_position_1 = feature_coords1[p][1]
        if(y_position_1 > 10 and x_position_1 > 10 and y_position_1 < height - 10 and x_position_1 < width - 10):
            for q in range(0, amount_points_2):
                y_position_2 = feature_coords2[q][0]
                x_position_2 = feature_coords2[q][1]
                if(y_position_2 > 7 and x_position_2 > 7 and y_position_2 < height - 7 and x_position_2 < width - 7):
                    window1 = image1[y_position_1 - 5: y_position_1 + 6, x_position_1 - 5: x_position_1 + 6]
                    window2 = image2[y_position_2 - 5: y_position_2 + 6, x_position_2 - 5: x_position_2 + 6]
                    sum_val = 0
                    for x in range(0,10):
                        for y in range(0,10):
                            val = window1[y,x] * window2[y,x]
                            sum_val += val
                    rank_1[p][q] = sum_val
    
    
    for p in range(0, amount_points_2):
        y_position_1 = feature_coords2[p][0]
        x_position_1 = feature_coords2[p][1]
        if(y_position_1 > 10 and x_position_1 > 10 and y_position_1 < height - 10 and x_position_1 < width - 10):
            for q in range(0, amount_points_1):
                y_position_2 = feature_coords1[q][0]
                x_position_2 = feature_coords1[q][1]
                if(y_position_2 > 10 and x_position_2 > 10 and y_position_2 < height - 10 and x_position_2 < width - 10):
                    window1 = image2[y_position_1 - 5: y_position_1 + 6, x_position_1 - 5: x_position_1 + 6]
                    window2 = image1[y_position_2 - 5: y_position_2 + 6, x_position_2 - 5: x_position_2 + 6]
                    sum_val = 0
                    for x in range(0,10):
                        for y in range(0,10):
                            val = window1[y,x] * window2[y,x]
                            sum_val += val
                    rank_2[p][q] = sum_val

    matches = []

    for l in range(0, 2):
        for h in range(0,amount_points_1): # find biggest NCC for that point
            ranks_point = rank_1[h:h+1,0:amount_points_2]
            highest_rank_position = np.argmax(ranks_point)
            ranks_marriage = rank_2[highest_rank_position:highest_rank_position+1,0:amount_points_2]
            highest_rank_position_2 = np.argmax(ranks_marriage)
            if (highest_rank_position_2 == h): # using the biggest NCC for the previous point, finds biggest NCC and checks if it is a mutual marriage
                matches.append((highest_rank_position_2, highest_rank_position))
                for k in range(0, amount_points_1):
                    rank_1[k][highest_rank_position] = 0
                for k in range(0, amount_points_2):
                    rank_2[k][highest_rank_position_2] = 0

    return matches



def mean_image(image): # function to subtract the mean of all pixels to each pixel and return that matrix

    (vertical, horizontal) = np.shape(image)

    mean_im = np.zeros((vertical,horizontal), dtype = "float32")

    mean_value = 0.0
    for x in range (0, horizontal): 
        for y in range(0, vertical):
            mean_value += float(image[y,x])
    mean_value = float(mean_value)/float(vertical * horizontal)
    for x in range (0, horizontal): 
        for y in range(0, vertical):
            mean_im[y][x] = float(image[y,x]) - mean_value
    
    return(mean_im, mean_value)



def variance_image(image, mean_value): # function to find the variance and divide each pixel by the variance
    (vertical, horizontal) = np.shape(image)

    variance_im = np.zeros((vertical,horizontal), dtype = "float32")

    variance_value = 0.0
    for x in range (0, horizontal): 
        for y in range(0, vertical):
            variance_value = float(variance_value) + (float(image[y,x])) ** 2
    variance_value = math.sqrt(float(variance_value)/float(vertical * horizontal))
    for x in range (0, horizontal): 
        for y in range(0, vertical):
            variance_im[y][x] = float(float(image[y,x])/float(variance_value))
    
    return(variance_im)

