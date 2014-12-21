"""
Course: COS 429
Author: David Fridovich-Keil

Use Kalman filter to guide matching on the next frame, given the matching
result from the previous frame.
"""

import numpy as np
import random, math
import cv2
import drawMatches
import pykalman

import time, sys, os

# file paths
IMGPATH = "../images/"
EXFILENAME = "orange_chinese.JPG"
VIDEOPATH = "../videos/orange_chinese/"
VIDEOBASENAME = "orange_chinese%4d.jpg"

# initialize kalman parameters
"""
ADD CODE HERE.
"""

# initialize color filter parameters
UPPERBOUND_ORANGE = 25
LOWERBOUND_ORANGE = 110
LOWERBOUND_LUM = 140
MEDIANSIZE = 3

# initialize other recurrent parameters
last_topleft = (None, None)
last_botright = (None, None)
CROPFACTOR = 1.5
frame = 1

# main loop
while os.path.isfile(VIDEOPATH + VIDEOBASENAME % frame):

    # read images
    test = cv2.imread(VIDEOPATH + VIDEOBASENAME % frame)
    ex = cv2.imread(IMGPATH + EXFILENAME)
    
    # convert to grayscale
    hsv_ex = cv2.cvtColor(ex, cv2.COLOR_RGB2HSV)
    hue_ex = hsv_ex[:,:,0]
    gray_ex = cv2.cvtColor(ex, cv2.COLOR_RGB2GRAY)


    hsv_test = cv2.cvtColor(test, cv2.COLOR_RGB2HSV)
    hue_test = hsv_test[:,:,0]
    gray_test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)

    # filter colors
    indices_ex = np.logical_or(np.logical_and(hue_ex > UPPERBOUND_ORANGE, 
                                              hue_ex < LOWERBOUND_ORANGE),
                               gray_ex > LOWERBOUND_LUM)
    gray_ex[indices_ex] = 255
    gray_ex[np.logical_not(indices_ex)] = 0

    indices_test = np.logical_or(np.logical_and(hue_test > UPPERBOUND_ORANGE, 
                                                hue_test < LOWERBOUND_ORANGE),
                                 gray_test > LOWERBOUND_LUM)
    gray_test[indices_test] = 255
    gray_test[np.logical_not(indices_test)] = 0

    gray_ex = cv2.medianBlur(gray_ex, MEDIANSIZE)
    gray_test = cv2.medianBlur(gray_test, MEDIANSIZE)

    # start timer
    starttime = time.time()

    # find ORB keypoints
    orb = cv2.ORB()
    kp_ex, des_ex = orb.detectAndCompute(gray_ex, None)
    kp_test, des_test = orb.detectAndCompute(gray_test, None)

    # do feature matching
    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING)
    matches_ex = bf.knnMatch(des_ex, des_test, k=2)

    # ratio test
    good_matches_ex = []
    for m,n in matches_ex:
        if m.distance < 0.8*n.distance:
            good_matches_ex.append(m)


    # halt if not enough good matches
    if len(good_matches_ex) < 4:
        print "Not enough good matches to estimate a homography."
        sys.exit()

    # estimate homography
    pts_ex = np.float32([kp_ex[m.queryIdx].pt 
                         for m in good_matches_ex]).reshape(-1,1,2)
    pts_test = np.float32([kp_test[m.trainIdx].pt 
                           for m in good_matches_ex]).reshape(-1,1,2)
    H_ex, mask_ex = cv2.findHomography(pts_ex, pts_test, cv2.RANSAC, 5.0)

    # draw boundary of ex code
    ex_h, ex_w = gray_ex.shape
    corners_ex = np.float32([[0, 0], [0, ex_h-1], 
                             [ex_w-1, ex_h-1], [ex_w-1, 0]]).reshape(-1,1,2)
    corners_test_ex = cv2.perspectiveTransform(corners_ex, H_ex)
    cv2.polylines(gray_test, [np.int32(corners_test_ex)], True, 120, 5)
    gray_test.shape

    # filter out inliers
    matchesMask_ex = mask_ex.ravel().tolist()
    inliers = []
    for i, m in enumerate(matchesMask_ex):
        if m == 1:
            inliers.append(good_matches_ex[i])

    # print elapsed time
    print "Total elapsed time for frame " + frame + ": " + \
        str(time.time() - starttime) + " seconds"
    frame = frame + 1

    # show images
    drawMatches.drawMatches(gray_ex, kp_ex, gray_test, kp_test, inliers) 
