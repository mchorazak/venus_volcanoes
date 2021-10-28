from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import logging
from datetime import timedelta
import helpers.config as conf
import helpers.data_structures as df
import sklearn.metrics as metrics

logger = logging.getLogger(conf.logger_name)


def run_ensemble(data):
    name = "ENSEMBLE"
    logger.info("________________________%s_______________________\n", name)

    estimators = []
    model1 = LogisticRegression(class_weight=data.class_weight)
    model1.max_iter = 2000
    estimators.append(('logistic1', model1))

    model2 = LogisticRegression(class_weight=data.class_weight)
    model2.max_iter = 2000
    estimators.append(('logistic2', model2))

    model3 = LogisticRegression(class_weight=data.class_weight)
    model3.max_iter = 2000
    estimators.append(('logistic4', model3))

    model4 = LogisticRegression(class_weight=data.class_weight)
    model4.max_iter = 2000
    estimators.append(('logistic3', model4))

    model5 = DecisionTreeClassifier(class_weight=data.class_weight,  max_leaf_nodes=40, max_depth=15)
    estimators.append(('dtc2', model5))

    model6 = DecisionTreeClassifier(class_weight=data.class_weight, max_leaf_nodes=40, max_depth=15)
    estimators.append(('dtc1', model6))

    model7 = DecisionTreeClassifier(class_weight=data.class_weight, max_leaf_nodes=40, max_depth=15)
    estimators.append(('dtc3', model7))

    model8 = DecisionTreeClassifier(class_weight=data.class_weight, max_leaf_nodes=40, max_depth=15)
    estimators.append(('dtc4', model8))

    ensemble = VotingClassifier(estimators)
    timer_start = time()
    ensemble.fit(data.x_train, data.y_train)
    timer_stop = time()
    training_time = (timer_stop - timer_start)
    logger.info('{} trained in: {}.'.format(name, timedelta(seconds=timer_stop-timer_start)))

    pred_test = ensemble.predict(data.x_test)
    pred_val = ensemble.predict(data.x_val)

    df.print_results(name, data.y_val, pred_val, data.y_test, pred_test)

    conf_matrix = metrics.confusion_matrix(data.y_test, pred_test)
    acc_score = metrics.accuracy_score(data.y_test, pred_test)
    class_rep = metrics.classification_report(data.y_test, pred_test)
    results = df.Results(conf_matrix, class_rep, acc_score, training_time, pred_test)

    return df.Model(name, results)
