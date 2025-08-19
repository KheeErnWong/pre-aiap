import logging
from .model import Model
from .datapipeline import Datapipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polyaxon")


def run_experiment(params, train_data_path, test_data_path):
    try:
        logger.info("Starting experiment")
        dpl = Datapipeline()

        # Load your data from your datapipeline
        X_train, y_train = dpl.transform_train_data(
            train_data_path
        )  # transform your train data using the train_data_path as a parameter

        X_test, y_test = dpl.transform_test_data(
            test_data_path
        )  # transform your test data using the test_data_path as a parameter

        model = Model()
        mse = model.train(params, X_train, y_train)
        test_mse = model.evaluate(X_test, y_test)

        # Log metrics to polyaxon
        logger.info(test_mse)
        logger.info("Experiment completed")

        return test_mse

    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")


if __name__ == "__main__":
    # Consider using something like argparse to get your arguments from the commandline and pass them as params to your model
    # For example, if using the argparse library,
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train_data_path')
    # parser.add_argument('--test_data_path')
    # Also ensure you set the default values to the best model you have

    params = {}
    run_experiment(params, train_data_path, test_data_path)
