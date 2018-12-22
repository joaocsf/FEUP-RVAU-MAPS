import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
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
  
  def mouse_click_callback(self, event, x, y, flags, param):
    if not event == cv.EVENT_LBUTTONDBLCLK:
      return
    self.database.associate_poi(self.selected_image_path, (x,y))
    drawnPOIS = self.draw_pois(self.selected_image_path, self.selected_image)
    cv.imshow('selected', drawnPOIS)
  
  def draw_pois(self, path, image):
    image = image.copy()
    pois = self.database.retrieve_pois(path)

    for k, v in pois.items():
      point = (v[0], v[1])
      cv.circle(image, point, 5, (0,255,0), thickness=-1)
      cv.putText(image, k, point, cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv.LINE_AA)

    return image

  def open_image(self, path):
    cv.namedWindow("selected", cv.WINDOW_NORMAL)
    self.selected_image = cv.imread(path, cv.IMREAD_COLOR)
    self.selected_image_path = path
    drawnPOIS = self.draw_pois(self.selected_image_path, self.selected_image)
    cv.imshow('selected', drawnPOIS)
    cv.setMouseCallback('selected', self.mouse_click_callback)
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

  def _update_poi_button(self):
    self.CYCLE_POI['state'] = tk.NORMAL if self.database.has_pois() else tk.DISABLED
    self.CYCLE_POI.config(text=self.database.get_poi_label())

  def _add_poi(self):
    string = tk.simpledialog.askstring('Point of Interest', 'Point of Interest Name')
    if string is None:
      return
    
    self.database.add_poi(string)
    self._update_poi_button()

  def _cycle_poi(self):
    self.database.next_poi()
    self._update_poi_button()

  def create_widgets(self):
    self.image_dependent_buttons = []

    self.QUIT = tk.Button()
    self.QUIT['text'] = "Quit"
    self.QUIT['fg'] = "red"
    self.QUIT['command'] = self.quit
    self.QUIT.grid(row=0, column=0)

    self.OPEN_IMG = tk.Button()
    self.OPEN_IMG['text'] = 'Open'
    self.OPEN_IMG['command'] = self.open_image_dialog
    self.OPEN_IMG.grid(row=0, column=1)

    self.CYCLE_PICTURES = tk.Button()
    self.CYCLE_PICTURES['state'] = tk.DISABLED
    self.CYCLE_PICTURES['text'] = 'Cycle Pictures'
    self.CYCLE_PICTURES['command'] = self._cycle_pictures
    self.CYCLE_PICTURES.grid(row=0, column=2)

    self.SHOW_KP = tk.Button()
    self.SHOW_KP['state'] = tk.DISABLED
    self.SHOW_KP['text'] = 'Show Features'
    self.SHOW_KP['command'] = self._show_keypoints
    self.SHOW_KP.grid(row=0, column=3)

    tk.Label(text='Points of Interest:').grid(row=1, column=0, columnspan=2)

    self.ADD_POI = tk.Button()
    self.ADD_POI['text'] = 'Add'
    self.ADD_POI['state'] = tk.DISABLED
    self.ADD_POI['command'] = self._add_poi
    self.ADD_POI.grid(row=2, column=0, columnspan=1)

    self.CYCLE_POI = tk.Button()
    self.CYCLE_POI['text'] = 'POI NAME'
    self.CYCLE_POI['state'] = tk.DISABLED
    self.CYCLE_POI['command'] = self._cycle_poi
    self.CYCLE_POI.grid(row=2, column=1, columnspan=2)

    self.image_dependent_buttons.append(self.SHOW_KP)
    self.image_dependent_buttons.append(self.ADD_POI)

  def __on_db_loaded__(self):
    length = len(self.database.retrieve_keys())
    self.CYCLE_PICTURES.config(
      state= tk.NORMAL if length > 0 else tk.DISABLED,
      text='Cycle Pictures: {0}/{1}'.format(self.current_selected_index, length))

    self._cycle_pictures() 
    self._update_poi_button()
    

  def __init__(self, master=None):
    tk.Frame.__init__(self, master)
    self.current_selected_index = 0
    self.database = Database('db.json')
    self.selected_image_path = None
    self.grid()
    self.create_widgets()
    self.__on_db_loaded__()


def main():
  root = tk.Tk()
  root.title('Toolbox')
  root.geometry('500x100')
  app = Application(master=root)
  app.mainloop()
  root.destroy()
  pass


if __name__ == "__main__":
  main()