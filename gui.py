import tkinter as tk 
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model
import os

# load model
model = load_model('modelRefine.h5')

# dictionary for bird classes
root = 'C:/Users/Reece/DataScience/birdImageClassif/birdDataSet/train'
classes = {}
for index,item in enumerate(os.listdir(root)):
    classes[index] = str(item)
    
# GUI
top = tk.Tk()
top.resizable(width=False, height=False)
top.minsize(width=800, height=600)
top.geometry = ('2000x2000')
top.title('Bird Image Classifier')
top.configure(background='#CDCDCD')
label = tk.Label(background='#CDCDCD', font=('helvetica',15,'bold'))
sign_image = tk.Label(top)

def classify(path):
    image = Image.open(path)
    image = image.resize((224,224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    prediction = model.predict_classes([image])[0]
    result = classes[prediction]
    print(result)
    label.configure(foreground='#000000', text=result)

def classify_button(path):
    button = tk.Button(top, text='Classify', command = lambda: classify(path),
    padx=10, pady=5)
    button.configure(background='#FF0000', foreground='white', font=('hevetica',10,'bold'))
    button.place(relx=0.8, rely=0.5)

def upload_img():
    try:
        path = filedialog.askopenfilename()
        uploaded = Image.open(path)
        uploaded.thumbnail(((top.winfo_width()/2),
        (top.winfo_height()/2)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        classify_button(path)
    except:
        pass

upload = tk.Button(top,text="Upload Image", command = upload_img, padx=10, pady=5)
upload.configure(background='#FF0000', foreground='white', font=('hevetica',10,'bold'))
upload.pack(side=tk.BOTTOM, pady=10)
sign_image.configure(background='#CDCDCD')
sign_image.pack(side=tk.BOTTOM, expand=True)
label.pack(side=tk.BOTTOM, expand=True)
heading = tk.Label(top, text='Bird Image Classifier', pady=20, font=('hevetica',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
    