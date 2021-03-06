%% Testing QR Code Tracking
% Author: Ankush Gola
% heavily adapted from http://www.mathworks.com/help/vision/examples/object-detection-in-a-cluttered-scene-using-point-feature-matching.html#btt5qyu
qr_scale = 0.5;
scene_scale = 0.5;
scene = 'scene_2.jpg';

qrImage = imresize(imread('qr.jpg'), qr_scale);

sceneImage = imresize(imread(scene), scene_scale);
sceneGray = rgb2gray(sceneImage);

qrPoints = detectSURFFeatures(qrImage);
scenePoints = detectSURFFeatures(sceneGray);

[qrFeatures, qrPoints] = extractFeatures(qrImage, qrPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneGray, scenePoints);

qrPairs = matchFeatures(qrFeatures, sceneFeatures);

matchedQRPoints = qrPoints(qrPairs(:, 1), :);
matchedScenePoints = scenePoints(qrPairs(:, 2), :);

[tform, inlierQRPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedQRPoints, matchedScenePoints, 'affine', ...
    'MaxNumTrials', 2000);

figure;
showMatchedFeatures(qrImage, sceneImage, inlierQRPoints, ...
    inlierScenePoints, 'montage');
title('Matched Points (Inliers Only)');

qrBound = [1, 1; ...
    size(qrImage, 2), 1; ...
    size(qrImage, 2), size(qrImage, 1); ...
    1, size(qrImage, 1); ...
    1, 1];

tQRBound = transformPointsForward(tform, qrBound);

figure;
imshow(sceneImage);
hold on;
line(tQRBound(:, 1), tQRBound(:, 2), 'Color', 'y');





