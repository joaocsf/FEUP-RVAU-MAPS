import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
from pprint import pprint
import pickle
import json
import os

class Database():
  def __init__(self, path):
    self.cached_features={}
    self.selected_poi=0
    self.dict = {
      'images': {},
      'poi': [],
      'index': 0}
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
    return [k for k in self.dict['images'].keys()]
  
  def add_poi(self, poi):
    self.dict['poi'].append(poi)
    self.selected_poi = self.dict['poi'].index(poi)
    self.save()
  
  def has_pois(self):
    return len(self.dict['poi']) != 0
  
  def get_poi(self):
    if len(self.dict['poi']) == 0: return None
    return self.dict['poi'][self.selected_poi]
  
  def get_poi_label(self):
    return '%s %d/%d'%(self.get_poi(), self.selected_poi+1, len(self.dict['poi']))
  
  def next_poi(self):
    self.selected_poi = (self.selected_poi + 1) % len(self.dict['poi'])
  
  def _retrieve_pois(self, path):
    images = self.retrieve_images_dict()
    obj = images[path]
    if not obj.__contains__('pois'):
      obj['pois'] = {}

    return obj['pois']
  
  def retrieve_pois(self, path):
    return self._retrieve_pois(path)
  
  def associate_poi(self, path, coordinates):
    pois = self._retrieve_pois(path)
    pois[self.get_poi()] = coordinates
    self.save()

  def retrieve_features(self, path):
    images = self.retrieve_images_dict()
    obj = images[path]
    features_filename = obj['features']
    if not self.cached_features.__contains__(features_filename):
      bundle = load_features(features_filename)
      self.cached_features[features_filename] = bundle

    return self.cached_features[features_filename]

  def _compute_features(self, path):
    images = self.retrieve_images_dict()
    obj = images[path]
    index = self.dict['index']
    if not os.path.exists('features/'):
      os.mkdir('features')

    features_filename = 'features/features_{0}.b'.format(index)
    obj['features']=features_filename
    self.dict['index'] = index + 1

    image = cv.imread(path, cv.IMREAD_COLOR)
    kp, des = calculate_key_points_akaze(image)
    store_features(features_filename, kp, des)

  def retrieve_images_dict(self):
    if not self.dict.__contains__('images'):
      self.dict['images']={}

    return self.dict['images']
  
  def file_exists(self, path):
    images = self.retrieve_images_dict()
    return (images.__contains__(path))

  def store_image(self, path):
    if len(path) == 0:
      return

    images = self.retrieve_images_dict()
    if(not self.file_exists(path)):
      images[path] = {}
    else:
      return

    self._compute_features(path)
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



