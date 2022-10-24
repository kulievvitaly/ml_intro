import tensorflow as tf
from models import *

import time, random, os
import cv2
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

import imgaug.augmenters as iaa

aug_seq_train = iaa.Sequential([
    iaa.Sometimes(0.95, iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            order=[0, 1],
            mode="edge" , cval=(0, 255)
        ) ),
])

BATCH_SIZE = 32

class_name_to_id_dict = {
    'airplane': 0,
    'bird': 1,
    'dog': 2,
    'horse': 3,
    'cat': 4,
    'automobile': 5,
    'frog': 6,
    'truck': 7,
    'ship': 8,
    'deer': 9,
}

def get_data_generator(test_flag):
    global_data_list = []
    # class_set = set()
    with open('data/trainLabels.csv') as f:
        f.readline()
        for s in f.readlines():
            # print('s', s)
            img_id, class_name = s[:-1].split(',')
            global_data_list.append((f'{img_id}.png', class_name_to_id_dict[class_name]))
            # class_set.add(class_name)
    # print('class_set', class_set)

    random.seed(0)
    random.shuffle(global_data_list)
    print('global_data_list', len(global_data_list), global_data_list[:10])

    data_list = []
    for count, item in enumerate(global_data_list):
        if (test_flag is False and count % 10 != 0) or \
            (test_flag is True and count % 10 == 0):
            data_list.append(item)

    print('data_list', len(data_list), test_flag)

    while True:
        X = []
        Y = []
        for i in range(BATCH_SIZE):
            fn, class_idx = random.choice(data_list)
            img_bgr = cv2.imread(f'data/train/{fn}')
            img_rgb = img_bgr[..., ::-1] / 255.
            if test_flag is False:
                img_rgb = aug_seq_train.augment_images([img_rgb])[0]
                img_rgb += np.ones(dtype=np.float64, shape=img_rgb.shape) * (np.random.random() * 2 - 1) * 0.05
                img_rgb += np.random.normal(0, 0.03, (img_rgb.shape))
            img_rgb = np.clip(img_rgb, 0, 1)

            X.append(img_rgb)
            y = [0 for _ in range(10)]
            y[class_idx] = 1
            Y.append(y)

        yield np.asarray(X), np.asarray(Y)


def show_generator_data(generator):
    while True:
        data = generator.__next__()
        imgs, Y = data
        print(Y[0], imgs[0].shape)
        cv2.imshow('img', imgs[0][..., ::-1])
        if cv2.waitKey() == 27:
            break


if __name__ == '__main__':
    model = get_model()
    # model.summary()

    generator_test = get_data_generator(True)
    generator_train = get_data_generator(False)

    # show_generator_data(generator_train)


    # lr = 0.0003
    # optimizer = optimizers.Adam(lr=lr)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # print(lr)
    #
    # callback = ModelCheckpoint('cifar_weights.h5', save_best_only=False, save_weights_only=True)
    #
    # model.fit_generator(generator_train, validation_data=generator_test, epochs=10000, verbose=2,
    #                     use_multiprocessing=True,
    #                     workers=4,
    #                     callbacks=[callback],
    #                     validation_steps=100,
    #                     steps_per_epoch=500,
    #                     )

    # model.load_weights('cifar_weights.h5')
    # while True:
    #     data = generator_test.__next__()
    #     imgs, Y = data
    #     print('-' * 50)
    #     print(Y[0])
    #
    #     predict = model.predict(imgs)[0]
    #     print('predict', predict)
    #
    #     cv2.imshow('img', imgs[0][..., ::-1])
    #     if cv2.waitKey() == 27:
    #         break


    model.load_weights('cifar_weights.h5')
    y_real = []
    y_predict = []
    for _ in range(100):
        data = generator_test.__next__()
        imgs, Y = data

        predicts = model.predict(imgs)
        for i in range(BATCH_SIZE):
            y_real.append(np.argmax(Y[i]))
            y_predict.append(np.argmax(predicts[i]))

    from sklearn import metrics
    import matplotlib.pyplot as plt
    confusion_matrix = metrics.confusion_matrix(y_real, y_predict)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.show()

