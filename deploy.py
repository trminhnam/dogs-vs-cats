import imghdr
import tkinter as tk
from tkinter import *
import numpy as np
import cv2
import tensorflow as tf
from PIL import ImageTk, Image
from tkinter import filedialog as fd 

model = tf.keras.models.load_model('CNN_0.88.h5')

categories = {
    0: 'cat',
    1: 'dog'
}

IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

window = tk.Tk()
window.maxsize(1920, 1080)
window.config(bg='white')
window.title('Cat and Dog classification')

label=Label(window, text='Cat and Dog Classification', background='white', font=('arial',20,'bold'))
label.grid(row=0, column=0, columnspan=2)

# Showing image area
drawing_area = Canvas(window, width=720, height=720, bg='white')
drawing_area.grid(row=1, column=0, padx=10, pady=5)

# Button area
button_area = Frame(window, width=100, height=380, bg='white')
button_area.grid(row=1, column=1, padx=10, pady=5)

upload_img_button = Button(button_area, text='Upload Image', command=lambda: upload_image(), width=20, height=2)
upload_img_button.grid(row=0, column=0, padx=5, pady=5)

classify_button = Label(button_area, text='', bg='white', width=20, height=2)
classify_button.grid(row=1, column=0, padx=5, pady=10)

prediction_position = Label(button_area, text='Cat', bg='white', font=('arial',15,'bold'))
prediction_position.grid(row=2, column=0, padx=5, pady=20)

img = None


def upload_image():
    # try:
        file_path=fd.askopenfilename()


        print(file_path)
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((window.winfo_width()/2.25), (window.winfo_height()/2.25)))
        
        
        global img
        img = ImageTk.PhotoImage(Image.open(file_path)) 
        
        # drawing_area = Canvas(window, bg='white', image=img)
        # drawing_area.grid(row=1, column=0, padx=10, pady=5)
        drawing_area.create_image(0+img.width()/2,0+img.height()/2,image=img)
        # drawing_area.update()

        label.configure(text='')
        # show_classify_button()
    # except:
        pass

def classify(img_path):
    # Load image
    image = cv2.imread(img_path)
    image = cv2.resize(image, dsize=IMG_SIZE)

    # Normalization
    image = image / 255

    # Classify
    pred = model.predict(np.array([image]))
    pred = np.array(tf.argmax(pred, axis=1))[0]
    
    # Output the result
    print(pred)
    sign = categories[pred]
    print(sign)

window.mainloop()
