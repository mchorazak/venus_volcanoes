from sklearn.svm import SVC
import sklearn.metrics as metrics
from time import time
from helpers.data_structures import print_results, Model, Results
from helpers import config
import logging
from datetime import timedelta

logger = logging.getLogger(config.logger_name)


def run_svc(data):
    # [[0 2223]
    #  [0  432]]
    # model = SVC(kernel='rbf',
    #             class_weight="balanced",
    #             max_iter=150)
    model = SVC(kernel='poly', degree=3, max_iter=500)


    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('SVC trained in: {}.'.format(timedelta(seconds=training_time)))

    # PREDICT
    pred_vali = model.predict(data.x_val)
    pred_test = model.predict(data.x_test)

    print_results(data.y_val, pred_vali, data.y_test, pred_test)

    results = Results(confusion_matrix(data.y_test, pred_test), classification_report(data.y_test, pred_test),
                      accuracy_score(data.y_test, pred_test), pred_test, training_time)
    return Model("SVM", results)

def run_svc1(data):
    # [[919 1304]
    #  [61  371]]
    # model = SVC(kernel='rbf',
    #             # class_weight="balanced",
    #             max_iter=150)
    model = SVC(kernel='poly', degree=5, max_iter=100)

    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('SVC trained in: {}.'.format(timedelta(seconds=training_time)))

    # PREDICT
    pred_vali = model.predict(data.x_val)
    pred_test = model.predict(data.x_test)

    print_results(data.y_val, pred_vali, data.y_test, pred_test)

    results = Results(confusion_matrix(data.y_test, pred_test), classification_report(data.y_test, pred_test),
                      accuracy_score(data.y_test, pred_test), pred_test, training_time)
    return Model("SVM1", results)


def run_svc2(data):
    # [[1681  542]
    #  [331  101]]
    # model = SVC(kernel='sigmoid',
    #             # class_weight="balanced",
    #             max_iter=150)

    model = SVC(kernel='poly', degree=10, max_iter=100)

    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('SVC trained in: {}.'.format(timedelta(seconds=training_time)))

    # PREDICT
    pred_vali = model.predict(data.x_val)
    pred_test = model.predict(data.x_test)

    print_results(data.y_val, pred_vali, data.y_test, pred_test)

    results = Results(confusion_matrix(data.y_test, pred_test), classification_report(data.y_test, pred_test),
                      accuracy_score(data.y_test, pred_test), pred_test, training_time)
    return Model("SVM2", results)

def run_svc3(data):
    # [[0 2223]
    #  [0  432]]
    # model = SVC(kernel='sigmoid',
    #             class_weight="balanced",
    #             max_iter=150)
    model = SVC(kernel='poly', degree=3, class_weight='balanced', max_iter=100)

    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('SVC trained in: {}.'.format(timedelta(seconds=training_time)))

    # PREDICT
    pred_vali = model.predict(data.x_val)
    pred_test = model.predict(data.x_test)

    print_results(data.y_val, pred_vali, data.y_test, pred_test)

    results = Results(metrics.confusion_matrix(data.y_test, pred_test), metrics.classification_report(data.y_test, pred_test),
                      metrics.accuracy_score(data.y_test, pred_test), pred_test, training_time)
    return Model("SVM3", results)