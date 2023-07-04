import tensorflow as tf



AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = [256, 256]

GCS_PATH = "Data"

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))

OUTPUT_CHANNELS = 3

EPOCH_SIZE = 25

model_path = 'Model/monet_generator.h5'


