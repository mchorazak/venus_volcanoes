from time import time
import logging
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
import helpers.config as conf
import helpers.data_structures as ds
import sklearn.metrics as metrics

logger = logging.getLogger(conf.logger_name)


def random_tree_forest(data):
    name = "RANDOM_FOREST"
    logger.info("_____________________%s_____________________\n", name)

    # TRAIN
    model = RandomForestClassifier(class_weight=data.class_weight, n_jobs=-1, n_estimators=1000, max_leaf_nodes=35, max_depth=15)
    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('{} trained in: {}.'.format(name, timedelta(seconds=timer_stop-timer_start)))

    # PREDICT
    pred_test = model.predict(data.x_test)
    pred_val = model.predict(data.x_val)

    ds.print_results(name, data.y_val, pred_val, data.y_test, pred_test)

    conf_matrix = metrics.confusion_matrix(data.y_test, pred_test)
    acc_score = metrics.accuracy_score(data.y_test, pred_test)
    class_rep = metrics.classification_report(data.y_test, pred_test)

    # PACKAGE RESULTS
    results = ds.Results(conf_matrix, class_rep, acc_score, training_time, pred_test)

    return ds.Model(name, results)
