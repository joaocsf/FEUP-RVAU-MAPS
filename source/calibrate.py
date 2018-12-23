import numpy as np
import cv2
import glob
import cv2
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Calibrate camera')

parser.add_argument('-cam', metavar='index', type=int, help='index of the camera', default=0)

args = parser.parse_args()

index = args.cam

print('Take pictures of multiple chessboards to calibrate\n-Space to take a picture\n-Esc to end')

cam = cv2.VideoCapture(int(index))

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Picture of chessboard", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(os.path.join('chessboards/', img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

## START CALIBRATION

X = 9
Y = 6
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((X*Y,3), np.float32)
objp[:,:2] = np.mgrid[0:Y,0:X].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('chessboards/opencv_frame_*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (Y,X),None)

    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (Y,X), corners,ret)
        cv2.imshow('Chessboard patterns',img)
        cv2.waitKey(1)

for fname in images:
    os.remove(fname)

cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# store mtx and dist