import file_output
from models.logistic_regression import run_logistic_regression
from models.cnn_short import run_cnn_short
from models.cnn_basic import run_cnn
from models.cnn_generated_data import run_cnn_generated_data
from models.decision_tree import run_decision_tree
from data_handling import get_data
from models.cnn_voting import cnn_voting
from models.random_tree_forest import random_tree_forest
import argparse
import logging
import helpers.config as conf
import cnn_analysis
from models.ensemble import run_ensemble
import model_comparison


def main():
    loaded_data = get_data()
    # CREATE AN ARRAY OF RESULTS
    models = []
    models.append(run_logistic_regression(loaded_data))

    models.append(run_decision_tree(loaded_data))
    models.append(random_tree_forest(loaded_data))
    models.append(run_ensemble(loaded_data))

    models.append(run_cnn(loaded_data))
    models.append(run_cnn(loaded_data))
    models.append(run_cnn_short(loaded_data))
    models.append(run_cnn_short(loaded_data))
    models.append(run_cnn_generated_data(loaded_data))
    models.append(run_cnn_generated_data(loaded_data))

    cnn_ens1, cnn_ens2 = cnn_voting(models, loaded_data)
    models.append(cnn_ens1)
    models.append(cnn_ens2)
    cnn_analysis.analyse_cnn(models)

    model_comparison.plot_models_results(models)

    if conf.write_out:
        file_output.print_results_to_file(models)
    logger.debug("Program exit.")


if __name__ == '__main__':
    logger = logging.getLogger(conf.logger_name)
    parser = argparse.ArgumentParser()
    # SET USER ARGUMENTS
    parser.add_argument('-p', action='store_false', default=True, help="flag to disable showing plots")
    parser.add_argument('-w', action='store_true', default=False,
                        help="flag to write result out to file. Default: do not write")
    parser.add_argument('-v', '--verbosity', default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Log level.")
    parser.add_argument('-c', '--cnn', type=int, default=2, choices=range(0, 3),
                        help="neural network training status: 0-silent, 1-progress bar, 2-medium")
    parser.add_argument('-e', '--epoch', type=int, default=30, help="Number of epochs for the CNN. Default: 30")
    parser.add_argument('-b', '--batch', type=int, default=64, help="Number of batches for a CNN epoch. Default: 64")

    # STORE USER ARGUMENTS
    args = parser.parse_args()
    conf.show_plots = args.p
    conf.cnn_verbosity = args.cnn
    conf.epochs = args.epoch
    conf.batch_size = args.batch
    conf.write_out = args.w
    logging.basicConfig(level=getattr(logging, args.verbosity), format='%(message)s')

    main()


