import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from PIL import Image
from shared import *
import json
from pprint import pprint
from sys import argv
import argparse
import math

index = 0

database = Database('db.json')
base = 0.5
height = 2
axis = np.float32([[-base,-base,0], [-base,base,0], [base,base,0], [base,-base,0],
                   [0,0,-height]])

compass = np.float32([[-0.2,0], [0.2,0], [0,1], [0,-1]])

poi_imgs = {}
debug = False
axis = axis*50
compass = compass*50

# Method to draw the compass given two points 
def draw_compass(image, p1, p2):
    # Calculating the vector performed by p1 and p2 points and calculate the rotation matrix to rotate the compass coordinates
    vector = (p2[0] - p1[0], p2[1] - p1[1])
    theta =np.arctan2(vector[0], -vector[1])
    c,s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s,c)))

    height, width = image.shape[:2]

    displaceX = 50
    displaceY = height - 50
    points = []
    # Apply the rotation matrix to each point stored on the compass array
    for p in compass:
        rp = R @ p
        rp = (int(rp[0]) + displaceX, int(rp[1]) + displaceY)
        points.append(rp)

    # Create the north and south shape and draw using the contours method
    first = np.array([points[0], points[1], points[2]])
    second = np.array([points[0], points[1], points[3]])

    cv.drawContours(image, [first], -1, (255,0,0), -1)
    cv.drawContours(image, [second], -1, (0,0, 255), -1)

# Method to draw the 3D pyamid at the poi point
def draw_axis(image, poi, rvec, tvec, H):
  global database, axis
  #Move all the corrdinates from the piramid by the poi coordinates
  axis1 = np.float32([ [x[0] + poi[0], x[1] + poi[1], x[2]] for x in axis])

  #Return the intrinsic matrix and distortion coeficientes and project the axis1 points 
  intrinsic, dist = database.get_calibration()
  points, _ = cv.projectPoints(axis1, rvec, tvec, intrinsic, dist)

  height, width = image.shape[:2]

  #Clamp the values to avoid problems when drawing the piramid
  points = [ [np.clip(x[0][0],0,width), np.clip(x[0][1], 0, height)]  for x in points]
  points = np.int32(points).reshape(-1,2)

  #Draw Base
  cv.drawContours(image, [points[:4]], -1, (0,100,10), -3)
  cv.drawContours(image, [points[:4]], -1, (0,200,10), 3)
  #Draw Edges
  for i in range(4):
    tmp = np.array([points[i], points[i+1], points[-1]])
    cv.drawContours(image, [tmp], -1, (0,200,10), -1)
    cv.line(image, tuple(points[i]), tuple(points[-1]),(0,255,0),3)

#Given an image, draw all the UI if there is a match on the database.
def evaluate(image):
    global database
    global debug
    global poi_imgs

    height, width, _ = image.shape
    centery = math.floor(height/2)
    centerx = math.floor(width/2)

    # draw crosshair
    cv.circle(image, (centerx,centery), 5, (0, 0, 0), thickness=-1)
    cv.circle(image, (centerx,centery), 3, (0, 255, 255), thickness=-1)

    # calculate the keypoints and descriptors using akaze
    features = calculate_key_points_akaze(image)

    # If no features are found, return the current image
    if features[1] is None:
        return image

    # Test every image inside the database until a match is found.
    # Returns T: Wich is the homography and camera transformation
    # POIS: The points of interest coordinates (from the matched image)
    # Scale: The scale measured in pixels/meters
    (T, pois, scale) = database.calculate_best_homogragy(features, image)

    if T is None or T[0] is None:
        return image

    homography, rvec, tvec = T

    poilist = []
    closest_i = 0
    closest_dist = 100000000000
    selected_poi = None
    poi_point = None

    # Invert the homography to canculate the center in the database's image space.
    _, transposed_homography = cv.invert(homography)
    transposed_center = homography_point(transposed_homography, (centerx, centery))

    # Calculate the closest POI
    for k, p in pois.items():
        # Compute the coordinates of the point in the new acquired image
        dp = homography_point(homography, p)
        dp = (int(dp[0]), int(dp[1]))
        poilist.append((k,dp))
        # Compute the distance in the database's image space
        distance = cv.norm((p[0], p[1]), transposed_center)
        if distance < closest_dist:
            selected_poi = p
            poi_point = dp
            closest_dist = distance
            closest_i = len(poilist)-1

    # Calculate the real distance
    real_distance = closest_dist/scale

    # Pre calculate the points of the compass around the current center
    p1 = homography_point(homography, (transposed_center[0], transposed_center[1] + 1000))
    p2 = homography_point(homography, (transposed_center[0], transposed_center[1] -1000))
    # Draw The compass given the points
    draw_compass(image, p1, p2)

    # Draw The pyramid
    draw_axis(image, selected_poi, rvec, tvec, homography)

    # Draw all the other poilist
    if debug:
        for k, dp in poilist:
            cv.putText(image, k, dp, cv.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 2, cv.LINE_AA)
            cv.circle(image, dp, 5, (0, 0, 0), thickness=-1)
            cv.circle(image, dp, 3, (0, 255, 0), thickness=-1)
            cv.drawKeypoints(image, features[0], image)
            if k == poilist[closest_i][0]:
                cv.line(image, dp, (centerx,centery),(0,0,255),thickness=3)
            else:
                cv.line(image, dp, (centerx,centery),(255,0,0), thickness = 2)

    # Compute the lower right corner square to display the stored image of the closest POI
    img_scale = 3
    imgx = math.floor(image.shape[1]/img_scale)
    imgy = math.floor(image.shape[0]/img_scale)

    uc = (image.shape[1]-10-imgx,image.shape[0]-10-30-imgy)
    lc = (image.shape[1],image.shape[0])

    textPos = (uc[0]+10, uc[1]+20)

    cv.line(image, poi_point, uc, (255,255,255), 2, lineType=cv.LINE_AA)
    cv.rectangle(image, uc, lc, (255,255,255), -1)

    cv.putText(image, '{0}: {1}'.format(poilist[closest_i][0], distance_to_text(real_distance)), textPos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)

    if poi_imgs[poilist[closest_i][0]] is not None:
        img = cv.resize(poi_imgs[poilist[closest_i][0]],(imgx,imgy))
        image[image.shape[0]-5-imgy:image.shape[0]-5,image.shape[1]-5-imgx:image.shape[1]-5] = img

    return image

#Method to open an image when a camera is not available
def open_file():
    filePath = tk.filedialog.askopenfilename(
        title='Select a Map\'s Image',
        filetypes=(
            ('all files', '*'),
            ('jpeg files', '*.jpg'),
            ('png files', '*.png')
        )
    )

    img = cv.imread(filePath, cv.IMREAD_COLOR)
    img = evaluate(img)
    cv.imshow('test', img)
    cv.waitKey(0)


#Method to execute the evaluate method in real time using a web camera
def real_time():
    cap = cv.VideoCapture(index)
    while True:
        ret, img = cap.read()

        img = evaluate(img)

        cv.imshow('test', img)
        if cv.waitKey(1) == 27:
            break

#Method to preload the POI images
def load_images():
    global poi_imgs
    global database
    pois = database.get_pois()
    for p in pois:
        img = cv.imread(p['image'])
        poi_imgs[p['name']] = img


#Main method to parse the program arguments
def main():
    global database
    parser = argparse.ArgumentParser(description='Calibrate camera')

    parser.add_argument('-cam', metavar='index', type=int,
                        help='index of the camera', default=0)

    parser.add_argument('-realtime', action='store_true')

    parser.add_argument('-debug', action='store_true')

    args = parser.parse_args()

    database.DEBUG = args.debug

    global index
    index = args.cam

    global debug
    debug = args.debug

    realtime = args.realtime

    load_images()

    if realtime:
        real_time()
    else:
        open_file()


if __name__ == "__main__":
    main()
