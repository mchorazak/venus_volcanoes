import os


def print_results_to_file(models):
    i = 0
    while os.path.exists("model_stats%s.csv" % i):
        i += 1

    file_object = open("model_stats%s.csv" % i, "w")
    # CREATE CSV LABELS
    labels = "model_name,accuracy,recall,cm_tn,cm_fp,cm_fn,cm_tp,time\n"
    file_object.write(labels)
    for model in models:
        conf_m = model.results.confusion_matrix
        recall = conf_m[1][1] / (conf_m[1][0] + conf_m[1][1])
        file_object.write(
            "{},{},{},{},{},{},{},{}\n".format(model.description, model.results.accuracy_score, recall, conf_m[0][0],
                                               conf_m[0][1], conf_m[1][0], conf_m[1][1], model.results.time))
    file_object.close()
