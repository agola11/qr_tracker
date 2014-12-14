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
QRFILENAME = "example.png"
BGFILENAME = "nassau.jpg"

# read images
bg = cv2.imread(IMGPATH + BGFILENAME)
bg = cv2.resize(bg, (int(round(bg.shape[1]*0.25)), 
                     int(round(bg.shape[0]*0.25))))

qr = cv2.imread(IMGPATH + QRFILENAME)
qr = cv2.resize(qr, (300, 300))

# create random composite
random.seed()
scale = random.uniform(0.25, 2.0)
scaled_qr = cv2.resize(qr, (int(round(scale*qr.shape[1])), 
                            int(round(scale*qr.shape[0]))))
paste_loc = (random.randint(0, bg.shape[1] - scaled_qr.shape[1] - 1), 
             random.randint(0, bg.shape[0] - scaled_qr.shape[0] - 1))
im = bg
im[paste_loc[1]:(paste_loc[1] + scaled_qr.shape[1]), 
   paste_loc[0]:(paste_loc[0] + scaled_qr.shape[0])] = scaled_qr

# convert to grayscale
gray_qr = cv2.cvtColor(qr, cv2.COLOR_RGB2GRAY)
gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

# start timer
starttime = time.time()

# find ORB keypoints
orb = cv2.ORB()
kp_qr, des_qr = orb.detectAndCompute(gray_qr, None)
kp_im, des_im = orb.detectAndCompute(gray_im, None)

# do feature matching
bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING)
matches = bf.knnMatch(des_qr, des_im, k=2)

# ratio test
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append(m)

# halt if not enough good matches
if len(good_matches) < 4:
    print "Not enough good matches to estimate a homography."
    sys.exit()

# estimate homography
pts_qr = np.float32([kp_qr[m.queryIdx].pt 
                     for m in good_matches]).reshape(-1,1,2)
pts_im = np.float32([kp_im[m.trainIdx].pt 
                     for m in good_matches]).reshape(-1,1,2)
H, mask = cv2.findHomography(pts_qr, pts_im, cv2.RANSAC, 5.0)

# draw boundary of qr code
qr_h, qr_w = gray_qr.shape
corners_qr = np.float32([[0, 0], [0, qr_h-1], 
                         [qr_w-1, qr_h-1], [qr_w-1, 0]]).reshape(-1,1,2)
corners_im = cv2.perspectiveTransform(corners_qr, H)
cv2.polylines(gray_im, [np.int32(corners_im)], True, 0, 5)
gray_im.shape

# filter out inliers
matchesMask = mask.ravel().tolist()
inliers = []
for i, m in enumerate(matchesMask):
    if m == 1:
        inliers.append(good_matches[i])

# print elapsed time
print "Total elapsed time: " + str(time.time() - starttime) + " seconds"

# show images
drawMatches.drawMatches(gray_qr, kp_qr, gray_im, kp_im, inliers) 
