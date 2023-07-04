import tensorflow as tf


from model import *
from CycleGan import CycleGan

# Introduction and Setup

# This notebook utilizes a CycleGAN architecture to add Monet-style to photos.
# For this tutorial, we will be using the TFRecord dataset. Import the following
# packages and change the accelerator to TPU.

# For more information, check out [TensorFlow](https://www.tensorflow.org/tutorials/generative/cyclegan)
# and [Keras](https://keras.io/examples/generative/cyclegan/) CycleGAN documentation
# pages.



# detect TPU device
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)




# We want to keep our photo dataset and our Monet dataset separate. First, load in the filenames of the TFRecords.
GCS_PATH = "Data"

# MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
# print('Monet TFRecord Files:', len(MONET_FILENAMES))

# PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
# print('Photo TFRecord Files:', len(PHOTO_FILENAMES))



MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_jpg/*.jpg'))
print('Monet TFRecord Files:', len(MONET_FILENAMES))

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_jpg/*.jpg'))
print('Photo TFRecord Files:', len(PHOTO_FILENAMES))


with strategy.scope():
    monet_generator = Generator() # transforms photos to Monet-esque paintings
    photo_generator = Generator() # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator() # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator() # differentiates real photos and generated photos


# Define loss functions

# The discriminator loss function below compares real images to a matrix of
# 1s and fake images to a matrix of 0s. The perfect discriminator will
# output all 1s for real images and all 0s for fake images. The discriminator
# loss outputs the average of the real and generated loss.

with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5
    


# The generator wants to fool the discriminator into thinking the generated image
# is real. The perfect generator will have the discriminator output only 1s.
# Thus, it compares the generated image to a matrix of 1s to find the loss.
with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)


# We want our original photo and the twice transformed photo to be
# similar to one another. Thus, we can calculate the cycle consistency
# loss be finding the average of their difference.
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1

# The identity loss compares the image with its generator
# (i.e. photo with photo generator). If given a photo as input, we want it
# to generate the same image as the image was originally a photo.
# The identity loss compares the input with the output of the generator.
with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss




# # Train the CycleGAN
# Let's compile our model. Since we used `tf.keras.Model` to build our
# CycleGAN, we can just ude the `fit` function to train our model.

with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


with strategy.scope():
    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )



cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=EPOCH_SIZE
)



# ! Model kaydediyoruz.
monet_generator.save(model_path)

