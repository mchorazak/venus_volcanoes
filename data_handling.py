from scipy.ndimage import gaussian_filter

import helpers.data_structures as ds
import helpers.config as conf
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from time import time
import os
from sklearn.model_selection import train_test_split
import logging
from datetime import timedelta
from collections import Counter

logger = logging.getLogger(conf.logger_name)


def get_data():
    logger.info("____________________DATA PROCESSING____________________\n")
    os.listdir('data/volcanoes_train/')

    # LOAD DATA FROM FILES
    logger.info("CSV loading...")
    timer_start = time()
    raw_train_images = pd.read_csv('data/volcanoes_train/train_images.csv', header=None)
    raw_train_labels = pd.read_csv('data/volcanoes_train/train_labels.csv')
    raw_test_images = pd.read_csv('data/volcanoes_test/test_images.csv', header=None)
    raw_test_labels = pd.read_csv('data/volcanoes_test/test_labels.csv')
    timer_stop = time()
    logger.info('CSV loaded in: {}.'.format(timedelta(seconds=timer_stop-timer_start)))

    # NORMALISE IMAGES AND DROP UNNECESSARY LABEL COLUMNS
    normalised_train_images = raw_train_images / 256
    normalised_train_labels = raw_train_labels['Volcano?']
    normalised_test_images = raw_test_images / 256
    normalised_test_labels = raw_test_labels['Volcano?']

    # SHOW SAMPLES OF RAW DATA
    if conf.show_plots:
        show_ratios(raw_train_labels, raw_test_labels)
        show_sample_images(raw_train_images, raw_train_labels)

    # DELETE CORRUPTED IMAGES
    logger.info("\nDeleting corrupted images...")
    timer_start = time()

    train_corrupted_list = get_index_corrupted_images(normalised_train_images)
    sum_deleted = len(train_corrupted_list)

    test_corrupted_list = get_index_corrupted_images(normalised_test_images)
    sum_deleted += len(test_corrupted_list)

    delete_corrupted_images(normalised_train_images, normalised_train_labels, train_corrupted_list)
    delete_corrupted_images(normalised_test_images, normalised_test_labels, test_corrupted_list)
    timer_stop = time()
    logger.info('Deleted {} corrupted images in: {}.'.format(sum_deleted, timedelta(seconds=timer_stop - timer_start)))

    x_train, x_val, y_train, y_val = train_test_split(normalised_train_images.values, normalised_train_labels.values, test_size=0.2, random_state=3)

    x_test = normalised_test_images.values
    y_test = normalised_test_labels.values

    # COMPUTE CLASS WEIGHTS TO HELP COMPENSATE FOR IMBALANCE
    counter = Counter(y_train)
    max_val = float(max(counter.values()))
    class_weight = {class_id: max_val / num_images for class_id, num_images in counter.items()}

    # RESHAPE TO SQUARE IMAGES, APPLY GAUSSIAN FILTER AND RESHAPE BACK
    img_width, img_height = 110, 110
    original_width = raw_train_images.shape[1]

    timer_start = time()

    x_train = x_train.reshape(-1, img_width, img_height, 1)
    x_test = x_test.reshape(-1, img_width, img_height, 1)
    x_val = x_val.reshape(-1, img_width, img_height, 1)

    x_train = gaussian_filter(x_train, sigma=0.12)
    x_test = gaussian_filter(x_test, sigma=0.12)
    x_val = gaussian_filter(x_val, sigma=0.12)

    x_train = x_train.reshape(-1, original_width)
    x_test = x_test.reshape(-1, original_width)
    x_val = x_val.reshape(-1, original_width)

    timer_stop = time()
    logger.info('Gaussian filters applied in: {}.'.format(timedelta(seconds=timer_stop - timer_start)))

    if conf.show_plots:
        show_single_image(x_train[0], "Gaussian filter")

    return ds.Data(x_train, y_train, x_val, y_val, x_test, y_test, class_weight)


def get_index_corrupted_images(data):
    corrupted_images_index = []
    # HORIZONTAL CHECK
    for i, image in enumerate(
            np.resize(data, (data.shape[0], 12100))):
        # CHECK TOP
        pix_sum = 0
        for pix_index in range(0, 110):
            pix_sum += image[pix_index]
            if pix_index == 10:
                break
        if pix_sum == 0:
            corrupted_images_index.append(i)
        # CHECK BOTTOM
        pix_sum = 0
        for pix_index in range(len(image)-110, len(image)):
            pix_sum += image[pix_index]
            if pix_index == 10:
                break
        if pix_sum == 0 and i not in corrupted_images_index:
            corrupted_images_index.append(i)

    # VERTICAL CHECK
    for i, image in enumerate(
            np.resize(data, (data.shape[0], 12100))):
        pix_sum = 0
        # CHECK LEFT SIDE
        for pix_index in range(0, len(image), 110):
            pix_sum += image[pix_index]
            if pix_index == 10:
                break
        if pix_sum == 0 and i not in corrupted_images_index:
            corrupted_images_index.append(i)

        # CHECK RIGHT SIDE
        pix_sum = 0
        for pix_index in range(110, len(image), 110):
            pix_sum += image[pix_index]
            if pix_index == 10:
                break
        if pix_sum == 0 and i not in corrupted_images_index:
            corrupted_images_index.append(i)

    return corrupted_images_index


def delete_corrupted_images(X, y, corrupted_index_list):
    for i in corrupted_index_list:
        X.drop(i, inplace=True)
        y.drop(i, inplace=True)

    X.reset_index(inplace=True)
    X.drop(['index'], axis=1, inplace=True)


def show_ratios(train_labels, test_labels):
    # SHOW DATA
    training_counter = train_labels['Volcano?'].value_counts()
    testing_counter = test_labels['Volcano?'].value_counts()
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    sb.barplot(training_counter.index, training_counter.values)
    plt.title('training')
    plt.subplot(122)
    sb.barplot(testing_counter.index, testing_counter.values)
    plt.title('testing')
    plt.show()


def show_sample_images(images, labels):
    # SHOW EXAMPLES
    volcano_images = images[labels['Volcano?'] == 1].sample(5)
    no_volcano_images = images[labels['Volcano?'] == 0].sample(5)

    plt.subplots(figsize=(15, 6))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(volcano_images.iloc[i, :].values.reshape((110, 110)), cmap='Greys')
        if i == 0:
            plt.ylabel('volcano')
    plt.show()

    plt.subplots(figsize=(15, 6))
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        if i == 0:
            plt.ylabel('no volcano')
        plt.imshow(no_volcano_images.iloc[i, :].values.reshape((110, 110)), cmap='Greys')
    plt.show()


def show_single_image(image, plot_label):
    plt.subplots(figsize=(5, 5))
    plt.imshow(image.reshape((110, 110)), cmap='Greys')
    plt.ylabel(plot_label)
    plt.show()


