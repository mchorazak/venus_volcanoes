from time import time
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import helpers.data_structures as ds
import helpers.config as conf
import logging
from datetime import timedelta

logger = logging.getLogger(conf.logger_name)


def run_decision_tree(data):
    name = "DECISION_TREE"
    logger.info("_____________________%s_____________________\n", name)

    # TRAIN
    model = DecisionTreeClassifier(class_weight=data.class_weight, max_leaf_nodes=40, max_depth=15)
    timer_start = time()
    model.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('{} trained in: {}.'.format(name, timedelta(seconds=training_time)))

    # PREDICT
    pred_test = model.predict(data.x_test)
    pred_val = model.predict(data.x_val)

    ds.print_results(name, data.y_val, pred_val, data.y_test, pred_test)

    # PACKAGE RESULTS
    results = ds.Results(metrics.confusion_matrix(data.y_test, pred_test), metrics.classification_report(data.y_test, pred_test),
                      metrics.accuracy_score(data.y_test, pred_test), training_time, pred_test)

    return ds.Model(name, results)
