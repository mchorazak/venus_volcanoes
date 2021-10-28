from time import time
import helpers.data_structures as ds
import helpers.config as conf
from tensorflow.keras.models import Sequential
import sklearn.metrics as metrics
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import logging
from datetime import timedelta

logger = logging.getLogger(conf.logger_name)


def run_cnn_short(data):
    name = "CNN_short"
    logger.info("________________________CNN_short______________________\n")

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
    model.add(Dropout(0.35))

    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(rate=0.35))
    model.add(Dense(1, activation="sigmoid"))

    # FIT MODEL
    logger.info("Fitting %s", name)
    timer_start = time()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(data.x_train, data.y_train, epochs=conf.epochs, batch_size=conf.batch_size,
                        validation_data=(data.x_val, data.y_val), verbose=conf.cnn_verbosity,
                        class_weight=data.class_weight)

    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('{} trained in: {}.'.format(name, timedelta(seconds=timer_stop-timer_start)))

    # PREDICT
    pred_val = model.predict_classes(data.x_val)
    pred_test = model.predict_classes(data.x_test)
    ds.print_results(name, data.y_val, pred_val, data.y_test, pred_test)

    result = ds.Results(metrics.confusion_matrix(data.y_test, pred_test), metrics.classification_report(data.y_test, pred_test),
                     metrics.accuracy_score(data.y_test, pred_test), training_time, pred_test, history)

    return ds.Model(name, result, tf_cnn=True)
