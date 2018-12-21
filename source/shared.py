import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk


def image_cv_2_tk(image):
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  pilFormat = Image.fromarray(image)
  return ImageTk.PhotoImage(pilFormat)