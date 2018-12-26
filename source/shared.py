import cv2 as cv
import numpy as np
from PIL import Image
from PIL import ImageTk
from pprint import pprint
import pickle
import json
import os

MIN_MATCH_COUNT = 5


class Database():
    def __init__(self, path):
        self.cached_features = {}
        self.selected_poi = 0
        self.calibration = load_camera_calibration('calibration.npy')
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
    def get_calibration(self):
      return self.calibration

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.dict, f)

    def retrieve_keys(self):
        return [k for k in self.dict['images'].keys()]

    def add_poi(self, poi):
        obj = {}
        obj['name'] = poi
        obj['image'] = ''
        self.dict['poi'].append(obj)
        self.selected_poi = len(self.dict['poi'])-1
        self.save()

    def add_poi_image(self, path):
        self.dict['poi'][self.selected_poi]['image'] = path
        self.save()

    def has_pois(self):
        return len(self.dict['poi']) != 0

    def get_poi(self):
        if len(self.dict['poi']) == 0:
            return None
        return self.dict['poi'][self.selected_poi]['name']

    def get_poi_image(self):
        if len(self.dict['poi']) == 0:
            return None
        return self.dict['poi'][self.selected_poi]['image']


    def get_pois(self):
        return self.dict['poi']

    def get_poi_label(self):
        return '%s %d/%d' % (self.get_poi(), self.selected_poi + 1, len(self.dict['poi']))

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

    def calculate_best_homogragy(self, image_features):
        someImage = self.retrieve_keys()[0]
        features = self.retrieve_features(someImage)
        T = compute_transformations_matrix(features, image_features, self.calibration[0], self.calibration[1])
        return (T, self.retrieve_pois(someImage))

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

    def retrieve_images_dict(self):
        if not self.dict.__contains__('images'):
            self.dict['images'] = {}

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

# Helper Methods


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


def store_features(path, keypoints, descriptors):
    array = features_to_pickle(keypoints, descriptors)
    pickle.dump(array, open(path, 'wb'))


def load_features(path):
    array = pickle.load(open(path, 'rb'))
    return pickle_to_features(array)

def store_camera_calibration(path, matrix, dist_coefs):
  np.save(path, np.array([matrix, dist_coefs]))

def load_camera_calibration(path):
  return np.load(path)

def homography_point(H, p):
  rp = H @ (p[0], p[1], 1)
  dp = (int(rp[0] / rp[2]), int(rp[1] / rp[2]))
  return dp

def compute_transformations_matrix(features1, features2, intrinsic_matrix, coef_points):
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(algorithm = FLANN_INDEX_LSH,
    #                  table_number = 6, # 12
    #                  key_size = 12,     # 20
    #                  multi_probe_level = 1)

    des1 = features1[1]
    des2 = features2[1]

    fb = cv.BFMatcher()

    # fb = cv.FlannBasedMatcher(index_params, search_params)
    # des1 = np.float32(des1)
    # des2 = np.float32(des2)

    matches = fb.knnMatch(des1, des2, k=2)

    good = []
    try:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
    except:
        return None
    if len(good) > MIN_MATCH_COUNT:
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
