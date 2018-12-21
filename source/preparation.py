import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
from PIL import Image
from shared import *
import json
from pprint import pprint


class Application(tk.Frame):
  
  def say_hi(self):
    print("Hello")
  
  def open_image_dialog(self):
    filePath = tk.filedialog.askopenfilename(
      title='Select a Map\'s Image',
      filetypes=(
        ('all files', '*'), 
        ('jpeg files', '*.jpg'), 
        ('png files','*.png')
        )
      )
    
    self.open_image(filePath)

  def click(self, event):
    print(event.x,event.y)
    text = "{0}, {1}".format(event.x, event.y)
    self.DEBUG.config(text=text)
  
  def _on_image_opened(self, opened):

    for btn in self.image_dependent_buttons:
      btn.config(state=tk.NORMAL if opened else tk.HIDDEN)

  def open_image(self, path):
    cv.namedWindow("selected", cv.WINDOW_NORMAL)
    self.selected_image = cv.imread(path, cv.IMREAD_COLOR)
    self.database.store_image(path)
    cv.imshow("selected", self.selected_image)
    self._on_image_opened(True)
  
  def calculate_features(self):
    kp, des = calculate_key_points_akaze(self.selected_image)
    return (kp, des)
  
  def _show_keypoints(self):
    kp, _ = self.calculate_features()
    self.show_keypoints(kp)
  
  def show_keypoints(self, keypoints):
    kp = keypoints
    img2 = self.selected_image.copy()
    cv.drawKeypoints(img2, kp,img2)
    self.DEBUG.config(text='Test..')
    cv.namedWindow('keypoints', cv.WINDOW_NORMAL)
    cv.imshow('keypoints', img2)
    self.DEBUG.config(text='Showing')
  
  def load_features(self):
    kp, _ = load_features('test.b')
    self.show_keypoints(kp)
    pass
  
  def store_features(self):
    kp, des = self.calculate_features()
    store_features('test.b', kp, des)
    pass
  
  def _cycle_pictures(self):
    keys = self.database.retrieve_keys()
    length = len(keys)

    self.CYCLE_PICTURES.config(
      state= tk.NORMAL if length > 0 else tk.DISABLED,
      text='Cycle Pictures: {0}/{1}'.format(self.current_selected_index+1, length))

    if(length > 0):
      self.open_image(keys[self.current_selected_index])
      self.current_selected_index = (self.current_selected_index + 1)%len(keys)

  def create_widgets(self):
    self.image_dependent_buttons = []

    self.QUIT = tk.Button()
    self.QUIT['text'] = "Quit"
    self.QUIT['fg'] = "red"
    self.QUIT['command'] = self.quit
    self.QUIT.pack(side=tk.LEFT)

    self.OPEN_IMG = tk.Button()
    self.OPEN_IMG['text'] = 'Open'
    self.OPEN_IMG['command'] = self.open_image_dialog
    self.OPEN_IMG.pack(side=tk.LEFT)

    self.CYCLE_PICTURES = tk.Button()
    self.CYCLE_PICTURES['state'] = tk.DISABLED
    self.CYCLE_PICTURES['text'] = 'Cycle Pictures'
    self.CYCLE_PICTURES['command'] = self._cycle_pictures
    self.CYCLE_PICTURES.pack(side=tk.LEFT)

    self.SHOW_KP = tk.Button()
    self.SHOW_KP['state'] = tk.DISABLED
    self.SHOW_KP['text'] = 'Key Points'
    self.SHOW_KP['command'] = self._show_keypoints
    self.SHOW_KP.pack(side=tk.LEFT)

    self.STORE_FEATURES = tk.Button()
    self.STORE_FEATURES['state'] = tk.DISABLED
    self.STORE_FEATURES['text'] = 'Store Features'
    self.STORE_FEATURES['command'] = self.store_features
    self.STORE_FEATURES.pack(side=tk.LEFT)

    self.LOAD_FEATURES = tk.Button()
    self.LOAD_FEATURES['state'] = tk.DISABLED
    self.LOAD_FEATURES['text'] = 'Load Features'
    self.LOAD_FEATURES['command'] = self.load_features
    self.LOAD_FEATURES.pack(side=tk.LEFT)

    self.image_dependent_buttons.append(self.SHOW_KP)
    self.image_dependent_buttons.append(self.STORE_FEATURES)
    self.image_dependent_buttons.append(self.LOAD_FEATURES)


    self.DEBUG = tk.Label()
    self.DEBUG.config(text='Test')
    self.DEBUG.pack(side=tk.BOTTOM)

  def __on_db_loaded__(self):
    length = len(self.database.retrieve_keys())
    self.CYCLE_PICTURES.config(
      state= tk.NORMAL if length > 0 else tk.DISABLED,
      text='Cycle Pictures: {0}/{1}'.format(self.current_selected_index, length))

    self._cycle_pictures() 
    

  def __init__(self, master=None):
    tk.Frame.__init__(self, master)
    self.current_selected_index = 0
    self.database = Database('db.json')
    self.pack()
    self.create_widgets()
    self.__on_db_loaded__()


def main():
  root = tk.Tk()
  root.title('Toolbox')
  root.geometry('500x50')
  app = Application(master=root)
  app.mainloop()
  root.destroy()

  pass


if __name__ == "__main__":
  main()