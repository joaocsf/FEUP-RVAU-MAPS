import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
from PIL import Image
from shared import *

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
    self.DEBUG.insert(INSERT, text)


  def open_image(self, path):
    image = cv.imread(path, cv.IMREAD_COLOR)
    tkImage = image_cv_2_tk(image)
    self.IMAGE_PANEL.configure(image=tkImage)
    self.IMAGE_PANEL.image = tkImage
    self.IMAGE_PANEL.bind('<Button-1>', self.click)

  def create_widgets(self):

    self.DEBUG = tk.Label()
    self.DEBUG.config(text='Test')
    self.DEBUG.pack(side=tk.TOP)

    self.IMAGE_PANEL = tk.Label()
    self.IMAGE_PANEL.pack(side=tk.LEFT)

    self.QUIT = tk.Button(self)
    self.QUIT['text'] = "Quit"
    self.QUIT['fg'] = "red"
    self.QUIT['command'] = self.quit
    self.QUIT.pack(side=tk.LEFT)

    self.OPEN_IMG = tk.Button(self)
    self.OPEN_IMG['text'] = 'Open'
    self.OPEN_IMG['command'] = self.open_image_dialog
    self.OPEN_IMG.pack(side=tk.LEFT)

    

    
  def __init__(self, master=None):
    tk.Frame.__init__(self, master)
    self.pack()
    self.create_widgets()


def main():
  root = tk.Tk()
  app = Application(master=root)
  app.mainloop()
  root.destroy()

  pass


if __name__ == "__main__":
  main()