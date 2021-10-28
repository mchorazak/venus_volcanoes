import matplotlib.pyplot as plt
import helpers.config as conf


def analyse_cnn(models):
    if conf.show_plots:
        plot_learning_curves_for_cnns(models)


def plot_learning_curves_for_cnns(models):
    cnn_models = [x for x in models if x.is_cnn]
    fig, ax = plt.subplots(2, len(cnn_models), figsize=(14, 6))
    for i, model in enumerate(cnn_models):
        ax[0][i].plot(model.results.history.epoch, model.results.history.history['loss'], color='b',
                      label="Training loss")
        ax[0][i].plot(model.results.history.epoch, model.results.history.history['val_loss'], color='r',
                      label="validation loss")
        ax[0][i].legend(loc='best', shadow=True)
        ax[0][i].set_title('loss vs epoch')

        ax[1][i].plot(model.results.history.epoch, model.results.history.history['accuracy'], color='b',
                      label="Training accuracy")
        ax[1][i].plot(model.results.history.epoch, model.results.history.history['val_accuracy'], color='r',
                      label="Validation accuracy")
        ax[1][i].legend(loc='best', shadow=True)
        ax[1][i].set_title('accuracy vs epoch')
    plt.show()
