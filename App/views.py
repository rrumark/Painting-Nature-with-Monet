from flask import redirect, render_template, request

from . import app, loaded_model, DOWNLOAD_FOLDER_INPUT, DOWNLOAD_FOLDER_OUTPUT, STATIC_DOWNLOAD_FOLDER_INPUT,  STATIC_DOWNLOAD_FOLDER_OUTPUT

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


def list_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_list.append(file_name)
    return file_list


@app.route("/", methods=["GET", "POST"])
def index_func():

    showKey = False

    # Kullınıcadan input dosyasını aldığımız zaman çalışacak. Yani post işlemi gerçekleşecek
    if request.method == "POST":

        showKey = True
        file = request.files["file"] # input değerinin ismini 
        input_image = file.read() # input dosyasını okuyoruz

        # inputu modele uygun hale getiriyorum
        input_image = process_input_image(input_image)
        input_image = tf.expand_dims(input_image, 0)

        # modelin içerisinde veriyorum 
        prediction = loaded_model(input_image, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8) # sonuç

        fileSize = len(list_files_in_folder(DOWNLOAD_FOLDER_OUTPUT)) + 1

        input_filename = f"input{fileSize}.jpg"
        output_filename = f"output{fileSize}.jpg"

        # input ve outpt dosyalarını kaydediyoruz
        output_image_path = os.path.join(DOWNLOAD_FOLDER_OUTPUT, output_filename)
        plt.imsave(output_image_path, prediction)

        input_image_path = os.path.join(DOWNLOAD_FOLDER_INPUT, input_filename)
        input_image = (input_image + 1) * 0.5
        plt.imsave(input_image_path, input_image[0].numpy())


        input_image_path = os.path.join(STATIC_DOWNLOAD_FOLDER_INPUT, input_filename)
        output_image_path = os.path.join(STATIC_DOWNLOAD_FOLDER_OUTPUT, output_filename)

        return render_template('index.html', showKey = showKey, input_image=input_image_path, output_image=output_image_path)

    return render_template('index.html', showKey = showKey)







@app.route("/history", methods=["GET", "POST"])
def history_func():


    inputFiles = sorted(list_files_in_folder(DOWNLOAD_FOLDER_INPUT))
    outputFiles = sorted(list_files_in_folder(DOWNLOAD_FOLDER_OUTPUT))
   

    return render_template("history.html",inputFiles = inputFiles, outputFiles = outputFiles)



















# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files["file"]

#         print(file)
#         return render_template('index.html')

#     return render_template('index.html')
