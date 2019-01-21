import os
os.getcwd()
import skimage
from skimage.transform import resize
from PIL import Image as IMG
import numpy as np

import time
import datetime
import shutil

import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.button import Button


from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.stacklayout import StackLayout
from kivy.uix.image import Image
from kivy.uix.popup import Popup


from kivy.config import Config
Config.set('kivy', 'desctop', 1)
Config.set('graphics', 'height', 700)
Config.set('graphics', 'width', 1000)

from mrcnnmodel import load_model
from mrcnn.visualize import save_image



class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    default_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    print(default_path)


class Root(StackLayout):
    
    # loading model before window open
    model = load_model()

    # default values for app
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    source = StringProperty(None)
    image_opacity = ObjectProperty(0)
    change_angle = ObjectProperty(0)

    # additional variable for side information 
    information = StringProperty("Количество деталей: 0")

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()
    
    def load(self, path, filename):
        print(path, filename)
        self.source = filename[0]
        self.image_opacity = 1
        self.dismiss_popup()



    def detection(self):


        class_names = ['bg', "Detail"] # names of classes for model
        
        # Optimizing image size before begin detection
        image = IMG.open(self.source)
        image = image.resize((1000, 800), IMG.ANTIALIAS)
        image = np.array(image)
        self.information = "Детекция начата"

        # Run detection
        t1 = time.time()
        results = self.model.detect([image], verbose=1)
        print('Detection time is {}'.format(time.time() - t1))
        r = results[0]
        self.information = ('Детекция заняла: {}'.format(time.time() - t1))
        
        # Making temporary file with masks and counting num of details
        pred_file = save_image(image, "temp", r['rois'], r['masks'], r['class_ids'],
                             r['scores'], class_names)

        # Show image inside the window and result of counting
        self.source = pred_file
        self.ids.mechimage.reload()
        self.num_of_details = len(r["class_ids"])
        self.information = "Количество деталей: " + str(self.num_of_details)


    def save(self):

        # Get current date
        # Create path to save detected image
        # Form name for detected file: time + number of details on image

        self.current_date = str(datetime.datetime.now().date())
        self.path_to_save = os.pardir + "\\output\\" + self.current_date

        old_name = "".join((self.path_to_save, "\\temp.jpg"))
        new_name = "".join((self.path_to_save, "\\", str(datetime.datetime.now().time()).replace(":","_")[:8], "_", str(self.num_of_details), "_details", ".jpg"))

        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

        shutil.copy(self.source, self.path_to_save)

        os.rename(old_name, new_name)
        self.information = "Сохранено!"
        os.remove(self.source)

class MechApp(App):

    def build(self):
        mechpart = Root()
        return mechpart


MechApp().run()

