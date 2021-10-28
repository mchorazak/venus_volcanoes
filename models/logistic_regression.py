from sklearn.linear_model import LogisticRegression
from time import time
import sklearn.metrics as metrics
import helpers.data_structures as ds
import helpers.config as conf
import logging
from datetime import timedelta

logger = logging.getLogger(conf.logger_name)

def run_logistic_regression(data):
    name = "LOGISTIC_REGRESSION"
    logger.info("__________________%s__________________\n", name)

    model = LogisticRegression(class_weight=data.class_weight)
    model.max_iter = 2000
    # TRAIN
    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('{} trained in: {}.'.format(name, timedelta(seconds=timer_stop-timer_start)))

    # PREDICT AND TEST
    pred_val = model.predict(data.x_val)
    pred_test = model.predict(data.x_test)

    ds.print_results(name, data.y_val, pred_val, data.y_test, pred_test)

    # PACKAGE RESULTS
    results = ds.Results(metrics.confusion_matrix(data.y_test, pred_test), metrics.classification_report(data.y_test, pred_test),
                      metrics.accuracy_score(data.y_test, pred_test), training_time, pred_test)

    return ds.Model(name, results)
