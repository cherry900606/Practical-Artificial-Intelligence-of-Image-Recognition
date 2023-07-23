from tensorflow import keras
import numpy as np
import cv2 as cv
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import urllib.request
import os


st.title("Please write your own words:")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # Fixed fill color with some opacity
    stroke_width=6,
    stroke_color="rgb(255, 255, 255)",
    background_color="rgb(0, 0, 0)",
    update_streamlit=True,
    height=150,
    width=150,
    drawing_mode="freedraw",
    point_display_radius=3,
    display_toolbar=True,
    key="full_app",
)



# model predict
if canvas_result.image_data is not None:
    if not os.path.isfile('model_2.h5'):
        urllib.request.urlretrieve('https://github.com/cherry900606/Practical-Artificial-Intelligence-of-Image-Recognition/raw/master/Final_Report/model_2.h5', 'model_2.h5')
    model = keras.models.load_model('model_2.h5', compile=False)
    drawing = cv.cvtColor(canvas_result.image_data, cv.COLOR_BGR2GRAY)
    cv.imshow('show',drawing)
    im = cv.resize(drawing, (32, 32))
    label = model.predict(im.reshape(1, 32, 32))
    pred = np.argmax(label)
    if pred >= 10:
        pred = chr(pred - 10 + ord('A')) 
    else:
        pred = chr(pred + ord('0'))

    st.title("Model prediction:")
    st.write(pred)
