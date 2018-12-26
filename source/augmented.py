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

index = 0

database = Database('db.json')


def evaluate(image):
    global database

    features = calculate_key_points_akaze(image)

    if features[1] is None:
        return image

    (homography, pois) = database.calculate_best_homogragy(features)

    if homography is None:
        return image

    for k, p in pois.items():
        rp = homography @ (p[0], p[1], 1)
        dp = (rp[0] / rp[2], rp[1] / rp[2])
        dp = (int(dp[0]), int(dp[1]))
        cv.putText(image, k, dp, cv.FONT_HERSHEY_SIMPLEX,
                   2, (0, 255, 0), 2, cv.LINE_AA)
        cv.circle(image, dp, 5, (0, 0, 255), thickness=-1)

    cv.drawKeypoints(image, features[0], image)
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

    args = parser.parse_args()

    global index
    index = args.cam

    realtime = args.realtime

    if realtime:
        real_time()
    else:
        open_file()


if __name__ == "__main__":
    main()
