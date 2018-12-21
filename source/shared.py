import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
from pprint import pprint
import pickle
import json

class Database():
  def __init__(self, path):
    self.dict = {'images': {}}
    self.path = path
    try:
      with open(path, 'r') as f:
        if f == None:
          return
        self.dict = json.load(f)
    except FileNotFoundError:
      print('File Not Found Creating new one')
      pass
  
  def save(self):
    with open(self.path, 'w') as f:
      json.dump(self.dict, f)

  def retrieve_keys(self):
    return self.dict['images'].keys()

  def store_image(self, path):
    if not self.dict.__contains__('images'):
      self.dict['images']={}

    images = self.dict['images']
    if(not images.__contains__(path)):
      images[path] = {}

    self.save()

def image_cv_2_tk(image):
  image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  pilFormat = Image.fromarray(image)
  return ImageTk.PhotoImage(pilFormat)

def calculate_key_points(image, detector):
  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  kp, des = detector.detectAndCompute(image, None)
  return (kp, des)

def calculate_key_points_sift(image):
  sift = cv.xfeatures2d.SIFT_create()
  return calculate_key_points(image, sift)

def calculate_key_points_surf(image):
  surf = cv.xfeatures2d.SURF_create(400)
  return calculate_key_points(image, surf)

def calculate_key_points_akaze(image):
  akaze = cv.AKAZE_create()
  return calculate_key_points(image, akaze)

def calculate_key_points_kaze(image):
  kaze = cv.KAZE_create()
  return calculate_key_points(image, kaze)

def features_to_pickle(keypoints, descriptors):
  array = []

  for index, kp in enumerate(keypoints):
    tmp = (
      kp.pt, kp.size, 
      kp.angle, kp.response, 
      kp.octave, kp.class_id, 
      descriptors[index])

    array.append(tmp)
  return array

def pickle_to_features(data):
  keypoints=[]
  descriptors=[]
  for point in data:
    kp = cv.KeyPoint(
      x=point[0][0], y=point[0][1],
      _size=point[1], _angle=point[2],
      _response=point[3], _octave=point[4],
      _class_id=point[5]
      )
    descriptor = point[6]
    keypoints.append(kp)
    descriptors.append(descriptor)
  return keypoints, np.array(descriptors)

def store_features(path, keypoints, descriptors):
  array = features_to_pickle(keypoints, descriptors)
  pickle.dump(array, open(path, 'wb'))

def load_features(path):
  array = pickle.load(open(path, 'rb'))
  return pickle_to_features(array)




if __name__ == "__main__":
    pass



