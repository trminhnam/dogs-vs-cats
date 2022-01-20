import tkinter as tk
from tkinter import *
import numpy as np
import cv2
import tensorflow as tf

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

# Showing image area
drawing_area = Canvas(window, width=720, height=720, bg='white')
drawing_area.grid(row=0, column=0, padx=10, pady=5)

# Button area
button_area = Frame(window, width=100, height=380, bg='white')
button_area.grid(row=0, column=1, padx=10, pady=5)

upload_img_button = Button(button_area, text='Upload Image', command=lambda: show_classify_button(), width=20, height=2)
upload_img_button.grid(row=0, column=0, padx=5, pady=5)

def show_classify_button():
    classify_button = Button(button_area, text='Classify', command=lambda x: max(x, 0), width=20, height=2)
    classify_button.grid(row=1, column=0, padx=5, pady=5)

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
