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

debug = False

def evaluate(image):
    global database
    global debug

    height, width, _ = image.shape
    centery = math.floor(height/2)
    centerx = math.floor(width/2)

    #draw crosshair
    cv.rectangle(image, (centerx-20,centery-2), (centerx+20,centery+2),(0,0,0),thickness=-1)
    cv.rectangle(image, (centerx-2,centery-20), (centerx+2,centery+20),(0,0,0),thickness=-1)
    cv.rectangle(image, (centerx-18,centery-0), (centerx+18,centery+0),(255,255,255),thickness=-1)
    cv.rectangle(image, (centerx-0,centery-18), (centerx+0,centery+18),(255,255,255),thickness=-1)

    features = calculate_key_points_akaze(image)

    if features[1] is None:
        return image

    (homography, pois) = database.calculate_best_homogragy(features)

    if homography is None:
        return image

    poilist = []
    closest_i = 0
    closest_dist = 100000000000
    for k, p in pois.items():
        rp = homography @ (p[0], p[1], 1)
        dp = (rp[0] / rp[2], rp[1] / rp[2])
        dp = (int(dp[0]), int(dp[1]))
        poilist.append((k,dp))
        distance = cv.norm(dp, (centerx,centery))
        if distance < closest_dist:
            closest_dist = distance
            closest_i = len(poilist)-1

    if debug:
        for k, dp in poilist:
            cv.putText(image, k, dp, cv.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 2, cv.LINE_AA)
            cv.circle(image, dp, 5, (0, 0, 255), thickness=-1)
            cv.drawKeypoints(image, features[0], image)
            if k == poilist[closest_i][0]:
                cv.line(image, dp, (centerx,centery),(0,0,255),thickness=3)
            else:
                cv.line(image, dp, (centerx,centery),(255,0,0), thickness = 2)
    else:
        k, dp = poilist[closest_i]
        cv.putText(image, k, dp, cv.FONT_HERSHEY_SIMPLEX,
                2, (0, 255, 0), 2, cv.LINE_AA)
        cv.circle(image, dp, 5, (0, 0, 255), thickness=-1)

    return image


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


def real_time():
    cap = cv.VideoCapture(index)
    while True:
        ret, img = cap.read()

        img = evaluate(img)

        cv.imshow('test', img)
        if cv.waitKey(1) == 27:
            break


def main():
    parser = argparse.ArgumentParser(description='Calibrate camera')

    parser.add_argument('-cam', metavar='index', type=int,
                        help='index of the camera', default=0)

    parser.add_argument('-realtime', action='store_true')

    parser.add_argument('-debug', action='store_true')

    args = parser.parse_args()

    global index
    index = args.cam

    global debug
    debug = args.debug

    realtime = args.realtime

    if realtime:
        real_time()
    else:
        open_file()


if __name__ == "__main__":
    main()
