import time, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)

TEXT = 'Red cat walking on the street'


if __name__ == '__main__':
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = keras_cv.models.StableDiffusion(jit_compile=True, img_height=256, img_width=256)

    images = model.text_to_image(TEXT, batch_size=2, num_steps=25)

    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()


