import logging
import sklearn.metrics as metrics
import numpy as np
import helpers.data_structures as ds
import helpers.config as conf
logger = logging.getLogger(conf.logger_name)


def cnn_voting(models, data):
    list_of_results = []
    # RETRIEVE PREDICTED VALUES FROM ALL CNN MODELS
    for model in (x for x in models if x.is_cnn):
        list_of_results.append(model.results.predicted)
    model_votes = np.concatenate(list_of_results, axis=1)

    # VOTING WHERE 1/3 OF VOTES CLASSIFY AS VOLCANO
    # THIS ENSEMBLE TRIES TO IMPROVE RECALL AT THE COST OF ACCURACY
    ensemble_vote = np.zeros(shape=(len(data.x_test), 1))
    for i, row in enumerate(model_votes):
        counts = np.sum(row)
        if counts / len(list_of_results) >= 1 / 3:
            ensemble_vote[i] = 1
        else:
            ensemble_vote[i] = 0

    # MAJORITY VOTING
    ensemble_vote2 = np.zeros(shape=(len(data.x_test), 1))
    for i, row in enumerate(model_votes):
        counts = np.bincount(row)
        ensemble_vote2[i] = np.argmax(counts)

    result = ds.Results(metrics.confusion_matrix(data.y_test, ensemble_vote), metrics.classification_report(data.y_test, ensemble_vote),
                     metrics.accuracy_score(data.y_test, ensemble_vote), -1, ensemble_vote)

    result2 = ds.Results(metrics.confusion_matrix(data.y_test, ensemble_vote2), metrics.classification_report(data.y_test, ensemble_vote2),
                      metrics.accuracy_score(data.y_test, ensemble_vote2), -1, ensemble_vote2)

    # TF_CNN EXPLICITLY FALSE BECAUSE NO "HISTORY" OBJECT AVAILABLE
    ensemble1 = ds.Model("CNN_ens_1/3", result, tf_cnn=False)
    ensemble2 = ds.Model("CNN_ens_maj", result2, tf_cnn=False)
    return ensemble1, ensemble2
