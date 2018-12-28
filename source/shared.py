import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
from pprint import pprint
import pickle
import json
import os

MIN_MATCH_COUNT = 10

# Database that contains features, points of interest, and images
# Features are stored in ./features
# Points of interest and map data is stored in db.json
class Database():
    def __init__(self, path):
        self.cached_features = {}
        self.selected_poi = 0
        try:
            self.calibration = load_camera_calibration('calibration.npy')
        except:
            print('Calibrate the camera first!')
            exit()
        self.dict = {
            'images': {},
            'main_image': None,
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
    # Get calibration
    def get_calibration(self):
      return self.calibration
    # Save data to json file
    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.dict, f)
    # Retrieve list of images
    def retrieve_keys(self):
        return [k for k in self.dict['images'].keys()]
    # Add new point of interest
    def add_poi(self, poi):
        obj = {}
        obj['name'] = poi
        obj['image'] = ''
        self.dict['poi'].append(obj)
        self.selected_poi = len(self.dict['poi'])-1
        self.save()
    # Add image to existing point of interest
    def add_poi_image(self, path):
        self.dict['poi'][self.selected_poi]['image'] = path
        self.save()
    # Check if the current selected image has any points of interest
    def has_pois(self):
        return len(self.dict['poi']) != 0
    # Get current selected POI (for preparation program)
    def get_poi(self):
        if len(self.dict['poi']) == 0:
            return None
        return self.dict['poi'][self.selected_poi]['name']
    # Get image of currently selected POI (for preparation program)
    def get_poi_image(self):
        if len(self.dict['poi']) == 0:
            return None
        return self.dict['poi'][self.selected_poi]['image']

    # Get list of POIs
    def get_pois(self):
        return self.dict['poi']
    # Get label of current POI
    def get_poi_label(self):
        return '%s %d/%d' % (self.get_poi(), self.selected_poi + 1, len(self.dict['poi']))
    # Change currently selected POI to next one (for preparation)
    def next_poi(self):
        self.selected_poi = (self.selected_poi + 1) % len(self.dict['poi'])

    def _retrieve_pois(self, path):
        images = self.retrieve_images_dict()
        obj = images[path]
        if not obj.__contains__('pois'):
            obj['pois'] = {}

        return obj['pois']
    # Retrieve list of pois on an image
    def retrieve_pois(self, path):
        return self._retrieve_pois(path)
    # Associate a created POI to an image
    def associate_poi(self, path, coordinates, poiName=None, update=True):
        pois = self._retrieve_pois(path)
        pois[self.get_poi() if poiName is None else poiName] = coordinates
        if update:
            self.update_key_points(path)
        self.save()
    # Set scale of a certain image
    def set_scale(self, path, scale):
        images = self.retrieve_images_dict()
        obj = images[path]
        obj['scale'] = scale

        self.save()
    # Get scale of certain image
    def get_scale(self, path):
        images = self.retrieve_images_dict()
        obj = images[path]
        value = None
        try:
            value = obj['scale']
        except:
            pass
        return obj['scale'] if not value is None else 1
    # Get the best homography between the features in 'image_features' and all images using SIFT
    def calculate_best_homogragy(self, image_features):

        for someImage in self.retrieve_keys():
            features = self.retrieve_features(someImage)
            T = compute_transformations_matrix(features, image_features, self.calibration[0], self.calibration[1])
            if T is None: continue
            H, rvec, tvec = T
            if not H is None:
                return (T, self.retrieve_pois(someImage), self.get_scale(someImage))

        return ((None, None, None), self.retrieve_pois(someImage), 0)
    # Retrieve features of a previously computed image
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
        obj['features'] = features_filename
        self.dict['index'] = index + 1

        image = cv.imread(path, cv.IMREAD_COLOR)
        kp, des = calculate_key_points_akaze(image)
        store_features(features_filename, kp, des)

    def update_key_points(self, control_image):
        control_features = self.retrieve_features(control_image)
        control_pois = self.retrieve_pois(control_image)

        img_keys = self.retrieve_keys()

        for key in img_keys:
            if key == control_image: continue
            image_features = self.retrieve_features(key)
            H = compute_homography(control_features, image_features)

            if H is None : continue
            
            for name, point in control_pois.items():
                new_point = homography_point(H, point)
                new_point = (int(new_point[0]), int(new_point[1]))
                self.associate_poi(key, new_point, name, False)
    # Get list of images
    def retrieve_images_dict(self):
        if not self.dict.__contains__('images'):
            self.dict['images'] = {}

        return self.dict['images']
    # Check if a map exists
    def file_exists(self, path):
        images = self.retrieve_images_dict()
        return (images.__contains__(path))
    # Store a map
    def store_image(self, path):
        if len(path) == 0:
            return

        images = self.retrieve_images_dict()
        if(not self.file_exists(path)):
            images[path] = {}
        else:
            return
        
        self._compute_features(path)

        if self.dict['main_image'] is None:
            self.dict['main_image'] = path
        else:
            self.update_key_points(self.dict['main_image'])

        self.save()

# Helper Methods


def image_cv_2_tk(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pilFormat = Image.fromarray(image)
    return ImageTk.PhotoImage(pilFormat)

# Compute the key points and descriptors using a certain detector
def calculate_key_points(image, detector):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(image, None)
    return (kp, des)

# Use SIFT to detect keypoints
def calculate_key_points_sift(image):
    sift = cv.xfeatures2d.SIFT_create()
    return calculate_key_points(image, sift)

# Use SURF to detect keypoints
def calculate_key_points_surf(image):
    surf = cv.xfeatures2d.SURF_create(400)
    return calculate_key_points(image, surf)

# Use AKAZE to detect keypoints
def calculate_key_points_akaze(image):
    akaze = cv.AKAZE_create()
    return calculate_key_points(image, akaze)

# Use KAZE to detect keypoints
def calculate_key_points_kaze(image):
    kaze = cv.KAZE_create()
    return calculate_key_points(image, kaze)

# Store keypoints and descriptors to file by serializing objects
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

# Store keypoints and descriptors to file by serializing objects
def pickle_to_features(data):
    keypoints = []
    descriptors = []
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

# Store keypoints and descriptors to file by serializing objects
def store_features(path, keypoints, descriptors):
    array = features_to_pickle(keypoints, descriptors)
    pickle.dump(array, open(path, 'wb'))

# Load keypoints and descriptors from file by deserializing objects
def load_features(path):
    array = pickle.load(open(path, 'rb'))
    return pickle_to_features(array)

# Store camera calibration matrix and distortion coefficients
def store_camera_calibration(path, matrix, dist_coefs):
  np.save(path, np.array([matrix, dist_coefs]))

# Load camera calibration matrix and distortion coefficients
def load_camera_calibration(path):
  return np.load(path)

def homography_point(H, p):
  rp = H @ (p[0], p[1], 1)
  dp = (rp[0] / rp[2], rp[1] / rp[2])
  return dp

# Compute homography using feature points
def compute_homography(features1, features2):
    des1 = features1[1]
    des2 = features2[1]

    fb = cv.BFMatcher()

    matches = fb.match(des1, des2)

   
    matches = sorted(matches, key = lambda x: x.distance)

    if len(matches) > MIN_MATCH_COUNT:
        good = matches[:MIN_MATCH_COUNT]
        pts1 = np.float32(
            [features1[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32(
            [features2[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)

        return H
    else:
        return None

def distance_to_text(distance):
    if distance >= 1000:
        return '{:.2f} km'.format(distance/1000)
    
    if distance > 100:
        return '{:.2f} km'.format(distance/1000)

    return '{:.2f} m'.format(distance)

# Use camera calibration matrix and distortion matrix to get homography and rotation/translation vectors
def compute_transformations_matrix(features1, features2, intrinsic_matrix, coef_points):
    des1 = features1[1]
    des2 = features2[1]

    fb = cv.BFMatcher()

    matches = fb.match(des1, des2)

    matches = sorted(matches, key = lambda x: x.distance)

    if len(matches) > MIN_MATCH_COUNT:
        good = matches[:MIN_MATCH_COUNT]
        pts1 = np.float32(
            [features1[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32(
            [features2[0][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        pts3 = []
        for pt in pts1:
          pts3.append([pt[0][0], pt[0][1], 0])

        pts3 = np.float32(pts3).reshape(-1,1,3)

        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)


        _, rvec, tvec, _= cv.solvePnPRansac(pts3, pts2,intrinsic_matrix, coef_points)

        return H, rvec, tvec
    else:
        return None


if __name__ == "__main__":
    pass
