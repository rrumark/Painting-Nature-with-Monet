from flask import redirect, render_template, request

from . import app, loaded_model, DOWNLOAD_FOLDER

from config import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


import os




def process_input_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [*IMAGE_SIZE])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image




@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        input_image = file.read()

        input_image = process_input_image(input_image)
        input_image = tf.expand_dims(input_image, 0)

        prediction = loaded_model(input_image, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

        output_image_path = os.path.join(DOWNLOAD_FOLDER, 'output.jpg')
        plt.imsave(output_image_path, prediction)

        input_image_path = os.path.join(DOWNLOAD_FOLDER, 'input.jpg')
        input_image = (input_image + 1) * 0.5
        plt.imsave(input_image_path, input_image[0].numpy())


        return render_template('index.html', input_image=input_image_path, output_image=output_image_path)

    return render_template('index.html')
































# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["file"]

#         print(file)
#         return render_template('index.html')

#     return render_template('index.html')
