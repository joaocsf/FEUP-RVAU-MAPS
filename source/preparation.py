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
    
    self.load_new_image(filePath)

  def click(self, event):
    print(event.x,event.y)
    text = "{0}, {1}".format(event.x, event.y)
    self.DEBUG.config(text=text)
  
  def _on_image_opened(self, opened):

    for btn in self.image_dependent_buttons:
      btn.config(state=tk.NORMAL if opened else tk.HIDDEN)

  def load_new_image(self, path):
    self.database.store_image(path)
    self.current_selected_index = self.database.retrieve_keys().index(path)
    self._cycle_pictures()

  def open_image(self, path):
    cv.namedWindow("selected", cv.WINDOW_NORMAL)
    self.selected_image = cv.imread(path, cv.IMREAD_COLOR)
    self.selected_image_path = path
    cv.imshow("selected", self.selected_image)
    self._on_image_opened(True)
  
  def _show_keypoints(self):
    kp, _ = self.database.retrieve_features(self.selected_image_path)
    self.show_keypoints(kp)
  
  def show_keypoints(self, keypoints):
    kp = keypoints
    img2 = self.selected_image.copy()
    cv.drawKeypoints(img2, kp,img2)
    cv.namedWindow('keypoints', cv.WINDOW_NORMAL)
    cv.imshow('keypoints', img2)
  
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
    self.SHOW_KP['text'] = 'Show Features'
    self.SHOW_KP['command'] = self._show_keypoints
    self.SHOW_KP.pack(side=tk.LEFT)

    self.image_dependent_buttons.append(self.SHOW_KP)

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
    self.selected_image_path = None
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