"""
Course: COS 429
Author: David Fridovich-Keil

Test feature matching using ORB features and RANSAC.
"""

import numpy as np
import random, math
import cv2
import drawMatches

import time, sys

IMGPATH = "../images/"
QRFILENAME = "qr_example.png"
ARFILENAME = "ar_example.png"
TESTFILENAME = "test1.jpg"

# read images
test = cv2.imread(IMGPATH + TESTFILENAME)
#test = cv2.resize(test, (int(round(test.shape[1]*0.25)), 
#                     int(round(test.shape[0]*0.25))))

qr = cv2.imread(IMGPATH + QRFILENAME)
qr = cv2.resize(qr, (300, 300))

ar = cv2.imread(IMGPATH + ARFILENAME)
ar = cv2.resize(ar, (300, 300))

# create random composite
#random.seed()
#scale = random.uniform(0.25, 2.0)
#scaled_qr = cv2.resize(qr, (int(round(scale*qr.shape[1])), 
#                            int(round(scale*qr.shape[0]))))
#paste_loc = (random.randint(0, test.shape[1] - scaled_qr.shape[1] - 1), 
#             random.randint(0, test.shape[0] - scaled_qr.shape[0] - 1))
#im = test
#im[paste_loc[1]:(paste_loc[1] + scaled_qr.shape[1]), 
#   paste_loc[0]:(paste_loc[0] + scaled_qr.shape[0])] = scaled_qr

# convert to grayscale
gray_qr = cv2.cvtColor(qr, cv2.COLOR_RGB2GRAY)
gray_test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)

# start timer
starttime = time.time()

# find ORB keypoints
orb = cv2.ORB()
kp_qr, des_qr = orb.detectAndCompute(gray_qr, None)
kp_test, des_test = orb.detectAndCompute(gray_test, None)

# do feature matching
bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING)
matches_qr = bf.knnMatch(des_qr, des_test, k=2)

# ratio test
good_matches_qr = []
for m,n in matches_qr:
    if m.distance < 0.75*n.distance:
        good_matches_qr.append(m)

# halt if not enough good matches
if len(good_matches_qr) < 4:
    print "Not enough good matches to estimate a homography."
    sys.exit()

# estimate homography
pts_qr = np.float32([kp_qr[m.queryIdx].pt 
                     for m in good_matches_qr]).reshape(-1,1,2)
pts_test = np.float32([kp_test[m.trainIdx].pt 
                     for m in good_matches_qr]).reshape(-1,1,2)
H_qr, mask_qr = cv2.findHomography(pts_qr, pts_test, cv2.RANSAC, 5.0)

# draw boundary of qr code
qr_h, qr_w = gray_qr.shape
corners_qr = np.float32([[0, 0], [0, qr_h-1], 
                         [qr_w-1, qr_h-1], [qr_w-1, 0]]).reshape(-1,1,2)
corners_test_qr = cv2.perspectiveTransform(corners_qr, H_qr)
cv2.polylines(gray_test, [np.int32(corners_test_qr)], True, 0, 5)
gray_test.shape

# filter out inliers
matchesMask_qr = mask_qr.ravel().tolist()
inliers = []
for i, m in enumerate(matchesMask):
    if m == 1:
        inliers.append(good_matches[i])

# print elapsed time
print "Total elapsed time: " + str(time.time() - starttime) + " seconds"

# show images
drawMatches.drawMatches(gray_qr, kp_qr, gray_test, kp_test, inliers) 
