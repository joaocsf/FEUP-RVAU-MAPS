import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from PIL import Image
from shared import *
import json
from pprint import pprint


def main():
  filePath = tk.filedialog.askopenfilename(
      title='Select a Map\'s Image',
      filetypes=(
        ('all files', '*'), 
        ('jpeg files', '*.jpg'), 
        ('png files','*.png')
        )
      )
  
  image = cv.imread(filePath, cv.IMREAD_COLOR)
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

  while cv.waitKey(1) != 27: continue

if __name__ == "__main__":
  main()