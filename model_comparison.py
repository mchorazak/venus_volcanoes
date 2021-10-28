import matplotlib.pyplot as plt
import pandas as pd


def plot_models_results(models):
    models_names = [x.description for x in models]
    accuracy = [x.results.accuracy_score for x in models]
    conf_mat = [x.results.confusion_matrix for x in models]
    models_recall = [(x[1][1]/(x[1][0]+x[1][1])) for x in conf_mat]

    df = pd.DataFrame({'accuracy': accuracy, 'recall': models_recall}, index=models_names)
    df.plot.bar(rot=45, grid=True)
    plt.show()
