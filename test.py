import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import *
from utils import *


loaded_model = tf.keras.models.load_model(model_path)

# Path to the input image
input_image_path = 'int/12.jpg'

def process_input_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [*IMAGE_SIZE])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

input_image = process_input_image(input_image_path)

# Making predictions by giving the image to the loaded model
input_image = tf.expand_dims(input_image, 0)  # Reshape to add batch size
prediction = loaded_model(input_image, training=False)[0].numpy()

# Convert estimate to range [0, 255]
prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

# Save image
output_image_path = 'Out/o1.jpg'
plt.imsave(output_image_path, prediction)

print(f"Prediction saved as {output_image_path}")


# Resmi yükle
input_image_ = mpimg.imread(input_image_path)

# Visualizing images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(input_image_)
ax[1].imshow(prediction)
ax[0].set_title("Input Photo")
ax[1].set_title("Monet-esque")
ax[0].axis("off")
ax[1].axis("off")

plt.show()
