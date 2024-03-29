import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
from PIL import Image
from shared import *
import json
from pprint import pprint
import math
import os
import shutil

#Classed used to store the preparation application state and UI
class Application(tk.Frame):

    #Debug method to test the interface
    def say_hi(self):
        print("Hello")

    #Method to store a new image to the database using a dialog
    def open_image_dialog(self):
        filePath = tk.filedialog.askopenfilename(
            title='Select a Map\'s Image',
            filetypes=(
                ('all files', '*'),
                ('jpeg files', '*.jpg'),
                ('png files', '*.png')
            )
        )

        self.load_new_image(filePath)

    #Method to receive click for debugging purposes
    def click(self, event):
        print(event.x, event.y)
        text = "{0}, {1}".format(event.x, event.y)
        self.DEBUG.config(text=text)

    #Observer for when an image is open (Button state is updated)
    def _on_image_opened(self, opened):

        for btn in self.image_dependent_buttons:
            btn.config(state=tk.NORMAL if opened else tk.HIDDEN)

    #Method to load a new image from the database, given an path
    def load_new_image(self, path):
        self.database.store_image(path)
        self.current_selected_index = self.database.retrieve_keys().index(path)
        self._cycle_pictures()

    #Mouse click callback for placing the POIS. Uses OPENCV window
    def mouse_click_callback(self, event, x, y, flags, param):
        if not event == cv.EVENT_LBUTTONDBLCLK:
            return
        self.database.associate_poi(self.selected_image_path, (x, y))
        drawnPOIS = self.draw_pois(
            self.selected_image_path, self.selected_image)
        cv.imshow(self.imageWindow, drawnPOIS)
        cv.waitKey(1)
    
    #Method to draw the POIS on the current image
    def draw_pois(self, path, image):
        image = image.copy()
        pois = self.database.retrieve_pois(path)

        for k, v in pois.items():
            point = (v[0], v[1])
            cv.putText(image, k, point, cv.FONT_HERSHEY_SIMPLEX,
                    2, (0, 170, 0), 2, cv.LINE_AA)
            cv.circle(image, point, 5, (0, 0, 0), thickness=-1)
            cv.circle(image, point, 3, (0, 170, 0), thickness=-1)
        poiimgpath = self.database.get_poi_image()

        return image

    #Main method to open an image, draws the POIS and updates the remaining buttons
    def open_image(self, path):
        self.selected_image = cv.imread(path, cv.IMREAD_COLOR)
        self.selected_image_path = path
        drawnPOIS = self.draw_pois(
            self.selected_image_path, self.selected_image)
        
        self.SET_SCALE.config(text='Scale:%f'%(self.database.get_scale(self.selected_image_path)))
        cv.imshow(self.imageWindow, drawnPOIS)
        cv.waitKey(1)
        self._on_image_opened(True)

    #Helper method to show the feature points for the current selected image
    def _show_keypoints(self):
        kp, _ = self.database.retrieve_features(self.selected_image_path)
        self.show_keypoints(kp)

    #Method to show the keypoints for the current selected image
    def show_keypoints(self, keypoints):
        kp = keypoints
        img2 = self.selected_image.copy()
        cv.drawKeypoints(img2, kp, img2)
        cv.namedWindow('keypoints', cv.WINDOW_NORMAL)
        cv.imshow('keypoints', img2)
        cv.waitKey(1)

    # Helper method to get the image of currently selected POI
    def _show_image(self):
        path = self.database.get_poi_image()
        self.show_image(path)

    # Method to show the image of currently selected POI
    def show_image(self, path):
        image = cv.imread(path)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.imshow('image', image)
        cv.waitKey(1)

    #Method to cycle the current selected picture from the database
    def _cycle_pictures(self):
        keys = self.database.retrieve_keys()
        length = len(keys)

        self.CYCLE_PICTURES.config(
            state=tk.NORMAL if length > 0 else tk.DISABLED,
            text='Cycle Pictures: {0}/{1}'.format(self.current_selected_index + 1, length))

        if(length > 0):
            self.open_image(keys[self.current_selected_index])
            self.current_selected_index = (
                self.current_selected_index + 1) % len(keys)

    #Method to update the point of interest button state
    def _update_poi_button(self):
        self.CYCLE_POI['state'] = tk.NORMAL if self.database.has_pois(
        ) else tk.DISABLED
        self.CYCLE_POI.config(text=self.database.get_poi_label())
    
    #Method to set the scale measured in Pixel/Meters 
    def _set_scale(self):
        value = tk.simpledialog.askfloat("Set Scale", 'Scale Format: Pixel/Distance (meters)', minvalue=0.0, maxvalue=999999.0)
        if value is None: 
            return
        self.database.set_scale(self.selected_image_path, value)
        self.SET_SCALE.config(text='Scale:%f'%(self.database.get_scale(self.selected_image_path)))

    #Helper Method to add a new Point of interest
    def _add_poi(self):
        string = tk.simpledialog.askstring(
            'Point of Interest', 'Point of Interest Name')
        if string is None:
            return

        self.database.add_poi(string)

        filePath = tk.filedialog.askopenfilename(
            title='Select a POI\'s Image',
            filetypes=(
                ('all files', '*'),
                ('jpeg files', '*.jpg'),
                ('png files', '*.png')
            )
        )
        self.database.add_poi_image(filePath)
    
    #Helper Method to update the image of a Point of interest
    def change_image(self):
        filePath = tk.filedialog.askopenfilename(
            title='Select a POI\'s Image',
            filetypes=(
                ('all files', '*'),
                ('jpeg files', '*.jpg'),
                ('png files', '*.png')
            )
        )
        self.database.add_poi_image(filePath)
        self._update_poi_button()
    
    #Method to remove the stored database and pre-computed features
    def quit2(self):
        try:
            os.remove('db.json')
            shutil.rmtree('features')
        except:
            pass
        self.quit()

    #Method to cycle the current selected poi
    def _cycle_poi(self):
        self.database.next_poi()
        self._update_poi_button()

    #Method to create all the widgets of the application
    def create_widgets(self):
        self.image_dependent_buttons = []

        self.imageWindow = 'selected'
        cv.namedWindow(self.imageWindow, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.imageWindow, self.mouse_click_callback)

        self.QUIT = tk.Button()
        self.QUIT['text'] = "Save and Quit"
        self.QUIT['fg'] = "red"
        self.QUIT['command'] = self.quit
        self.QUIT.grid(row=0, column=0)

        self.QUIT2 = tk.Button()
        self.QUIT2['text'] = "Clear and Quit"
        self.QUIT2['fg'] = "red"
        self.QUIT2['command'] = self.quit2
        self.QUIT2.grid(row=0, column=1)

        self.OPEN_IMG = tk.Button()
        self.OPEN_IMG['text'] = 'Open'
        self.OPEN_IMG['command'] = self.open_image_dialog
        self.OPEN_IMG.grid(row=0, column=2)

        self.CYCLE_PICTURES = tk.Button()
        self.CYCLE_PICTURES['state'] = tk.DISABLED
        self.CYCLE_PICTURES['text'] = 'Cycle Pictures'
        self.CYCLE_PICTURES['command'] = self._cycle_pictures
        self.CYCLE_PICTURES.grid(row=0, column=3)

        self.SHOW_KP = tk.Button()
        self.SHOW_KP['state'] = tk.DISABLED
        self.SHOW_KP['text'] = 'Show Features'
        self.SHOW_KP['command'] = self._show_keypoints
        self.SHOW_KP.grid(row=0, column=4)

        self.SET_SCALE = tk.Button()
        self.SET_SCALE['state'] = tk.DISABLED
        self.SET_SCALE['text'] = 'Set Scale'
        self.SET_SCALE['command'] = self._set_scale
        self.SET_SCALE.grid(row=0, column=5)

        tk.Label(text='Points of Interest:').grid(
            row=1, column=0, columnspan=2)

        self.ADD_POI = tk.Button()
        self.ADD_POI['text'] = 'Add'
        self.ADD_POI['state'] = tk.DISABLED
        self.ADD_POI['command'] = self._add_poi
        self.ADD_POI.grid(row=2, column=0, columnspan=1)

        self.IMAGE_POI = tk.Button()
        self.IMAGE_POI['text'] = 'Show image'
        self.IMAGE_POI['state'] = tk.DISABLED
        self.IMAGE_POI['command'] = self._show_image
        self.IMAGE_POI.grid(row=2, column=1, columnspan=1)

        self.IMAGE_POI_CHANGE = tk.Button()
        self.IMAGE_POI_CHANGE['text'] = 'Change image'
        self.IMAGE_POI_CHANGE['state'] = tk.DISABLED
        self.IMAGE_POI_CHANGE['command'] = self.change_image
        self.IMAGE_POI_CHANGE.grid(row=2, column=2, columnspan=1)

        self.CYCLE_POI = tk.Button()
        self.CYCLE_POI['text'] = 'POI NAME'
        self.CYCLE_POI['state'] = tk.DISABLED
        self.CYCLE_POI['command'] = self._cycle_poi
        self.CYCLE_POI.grid(row=2, column=3, columnspan=1)

        self.image_dependent_buttons.append(self.SHOW_KP)
        self.image_dependent_buttons.append(self.ADD_POI)
        self.image_dependent_buttons.append(self.IMAGE_POI)
        self.image_dependent_buttons.append(self.IMAGE_POI_CHANGE)
        self.image_dependent_buttons.append(self.SET_SCALE)

    #Event called when the database is loaded 
    def __on_db_loaded__(self):
        length = len(self.database.retrieve_keys())
        self.CYCLE_PICTURES.config(
            state=tk.NORMAL if length > 0 else tk.DISABLED,
            text='Cycle Pictures: {0}/{1}'.format(self.current_selected_index, length))

        self._cycle_pictures()
        self._update_poi_button()

    #Class Constructor, Reload or Create a new state for the application.
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.current_selected_index = 0
        self.database = Database('db.json')
        self.selected_image_path = None
        self.grid()
        self.create_widgets()
        self.__on_db_loaded__()


#Main method to create the application
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
