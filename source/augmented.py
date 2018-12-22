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


def evaluate(image):
  database = Database('db.json')

  features = calculate_key_points_akaze(image)
  pprint(features[1])
  (homography, pois) = database.calculate_best_homogragy(features)

  for k, p in pois.items():
    rp = homography @ (p[0],p[1],1)
    dp = (rp[0]/rp[2], rp[1]/rp[2])
    dp = (int(dp[0]), int(dp[1]))
    pprint(dp)
    cv.putText(image, k, dp, cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv.LINE_AA)
    cv.circle(image, dp, 5, (0,0,255), thickness=-1)

  cv.drawKeypoints(image, features[0], image)
  cv.imshow('result', image)

def open_file():
  filePath = tk.filedialog.askopenfilename(
      title='Select a Map\'s Image',
      filetypes=(
        ('all files', '*'), 
        ('jpeg files', '*.jpg'), 
        ('png files','*.png')
        )
      )

  img = cv.imread(filePath, cv.IMREAD_COLOR)
  evaluate(img)

def real_time():
  cap = cv.VideoCapture(1)
  while True:
    ret, img = cap.read()
    cv.imshow('test', ret)
    try:
      evaluate(img)
      pass
    except cv.error:
      pass
    
    if cv.waitKey(1) == 27: break


def main():
  if argv[1] == '-real-time':
    real_time()
  else:
    open_file()

if __name__ == "__main__":
  main()