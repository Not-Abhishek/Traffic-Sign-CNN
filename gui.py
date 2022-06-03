import tkinter as tk
from tkinter import *
from tkinter import filedialog

import numpy as np
import pyttsx3
from PIL import ImageTk, Image
from keras.models import load_model


class ClassesEnum(enum.Enum):
    classes = {1: 'Speed limit (20km/h)',
               2: 'Speed limit (30km/h)',
               3: 'Speed limit (50km/h)',
               4: 'Speed limit (60km/h)',
               5: 'Speed limit (70km/h)',
               6: 'Speed limit (80km/h)',
               7: 'End of speed limit (80km/h)',
               8: 'Speed limit (100km/h)',
               9: 'Speed limit (120km/h)',
               10: 'No passing',
               11: 'No passing veh over 3.5 tons',
               12: 'Right-of-way at intersection',
               13: 'Priority road',
               14: 'Yield',
               15: 'Stop',
               16: 'No vehicles',
               17: 'Veh > 3.5 tons prohibited',
               18: 'No entry',
               19: 'General caution',
               20: 'Dangerous curve left',
               21: 'Dangerous curve right',
               22: 'Double curve',
               23: 'Bumpy road',
               24: 'Slippery road',
               25: 'Road narrows on the right',
               26: 'Road work',
               27: 'Traffic signals',
               28: 'Pedestrians',
               29: 'Children crossing',
               30: 'Bicycles crossing',
               31: 'Beware of ice/snow',
               32: 'Wild animals crossing',
               33: 'End speed + passing limits',
               34: 'Turn right ahead',
               35: 'Turn left ahead',
               36: 'Ahead only',
               37: 'Go straight or right',
               38: 'Go straight or left',
               39: 'Keep right',
               40: 'Keep left',
               41: 'Roundabout mandatory',
               42: 'End of no passing',
               43: 'End no passing veh > 3.5 tons'}


class TsrGui:
    def __init__(self, main_tk):
        self.upload = Button(main_tk, text="Upload an image", command=self.upload_image, padx=10, pady=5,
                             foreground='#EDF2F4')
        self.upload.configure(background='#8C909B', foreground='#EDF2F4', font=('arial', 10, 'bold'))

        self.label = Label(main_tk, background='#2B2D42', font=('arial', 15, 'bold'))
        self.img_pred = Label(main_tk)
        self.upload.pack(side=BOTTOM, pady=50)
        self.img_pred.pack(side=BOTTOM, expand=True)
        self.label.pack(side=BOTTOM, expand=True)
        self.heading = Label(main_tk, text="Traffic Sign Recognition", pady=20, font=('arial', 20, 'bold'),
                             foreground='#EDF2F4')
        self.heading.configure(background='#2B2D42', foreground='#EDF2F4')
        self.heading.pack()
        self.model = load_model('tsr_model.h5')
        self.t2s = None
        self.file_path = None

    def text_2_speech(self):
        t2s = self.t2s + ' sign ahead'

        engine = pyttsx3.init()
        engine.setProperty("rate", 200)

        engine.say(t2s)
        engine.runAndWait()
        engine.stop()

    def model_predict(self):
        image = Image.open(self.file_path)
        image = image.resize((30, 30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred = self.model.predict([image])[0]
        classes_x = np.argmax(pred, axis=-1)
        sign = ClassesEnum.classes.value[classes_x + 1]
        print('\n', sign)
        self.label.configure(foreground='#EDF2F4', text=sign)
        self.t2s = sign
        self.text_2_speech()

    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        uploaded = Image.open(self.file_path)
        uploaded.thumbnail(((main_tk.winfo_width() / 2.25), (main_tk.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        self.img_pred.configure(image=im)
        self.img_pred.image = im
        self.label.configure(text='')
        self.model_predict()


if __name__ == '__main__':
    main_tk = tk.Tk()
    main_tk.geometry('800x600')
    main_tk.title('Traffic sign classification')
    main_tk.configure(background='#2B2D42')
    TsrGui(main_tk)
    main_tk.mainloop()
