from time import time
import helpers.data_structures as ds
import helpers.config as conf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import sklearn.metrics as metrics
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import logging
import matplotlib.pyplot as plt
from datetime import timedelta

logger = logging.getLogger(conf.logger_name)


def run_cnn_generated_data(data):
    name = "CNN_generated_data"
    logger.info("___________________%s__________________\n", name)

    # RESHAPE
    img_width, img_height = 110, 110
    data.x_train = data.x_train.reshape(-1, img_width, img_height, 1)
    data.x_test = data.x_test.reshape(-1, img_width, img_height, 1)
    data.x_val = data.x_val.reshape(-1, img_width, img_height, 1)

    # CONSTRUCT MODEL
    model = Sequential()

    model.add(Conv2D(6, (4, 4), activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(96, (4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))

    model.add(Conv2D(96, (4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(96, (4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1, activation="sigmoid"))

    # FIT MODEL
    timer_start = time()

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=30,  # RANDOM ROTATION UP TO 30 DEGREES
        zoom_range=False,
        width_shift_range=False,
        height_shift_range=False,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode="wrap")  # WRAPPING IMAGES

    datagen.fit(data.x_train)

    if conf.show_plots:
        for X_batch, y_batch in datagen.flow(data.x_train, data.y_train, batch_size=9):
            for i in range(0, 9):
                plt.subplot(330 + 1 + i)
                plt.imshow(X_batch[i].reshape(110, 110), cmap=plt.get_cmap('gray'))
                plt.ylabel("Generated data")
            plt.show()
            break

    generator = datagen.flow(data.x_train, data.y_train, batch_size=conf.batch_size)
    logger.info("Fitting %s...", name)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(generator, epochs=conf.epochs, validation_data=(data.x_val, data.y_val),
                        class_weight=data.class_weight,
                        steps_per_epoch=len(data.x_train) // conf.batch_size,
                        verbose=conf.cnn_verbosity)

    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('{} trained in: {}.'.format(name, timedelta(seconds=timer_stop - timer_start)))

    # PREDICT
    pred_val = model.predict_classes(data.x_val)
    pred_test = model.predict_classes(data.x_test)
    ds.print_results(name, data.y_val, pred_val, data.y_test, pred_test)

    results = ds.Results(metrics.confusion_matrix(data.y_test, pred_test),
                         metrics.classification_report(data.y_test, pred_test),
                         metrics.accuracy_score(data.y_test, pred_test), training_time, pred_test, history)

    return ds.Model(name, results, tf_cnn=True)
