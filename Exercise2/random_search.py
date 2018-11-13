import logging

logging.basicConfig(level=logging.WARNING)
import hpbandster
import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import argparse

import cnn_mnistCheck as mnist


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist.mnist("./")
        print('Test')

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filtersize = config["filtersize"]
        epochs = budget

        # TODO: train and validate your convolutional neural networks here
        tmp, tmp1, validation_error = mnist.train_and_validate(self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test, epochs, lr, num_filters, batch_size, filtersize)
        print({
            'loss': float(validation_error),  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })
       # TODO: We minimize so make sure you return the validation error here
        
        return ({
            'loss': float(validation_error),  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        
        configspace = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('learning_rate', lower=0.0001, upper=0.1, default_value='1e-2', log=True)
        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=2**3, upper=2**6, default_value=16, log=True)
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=128, log=True)
        filtersize = CSH.CategoricalHyperparameter('filtersize', [5,3])

        configspace.add_hyperparameters([lr, num_filters, batch_size, filtersize])

        return configspace


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()
print('step1')
# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)
print('step2')
# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)
print('step3')
# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()
print('step4')
# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
print(id2config)
print('Test')
print('Best found configuration:', id2config[incumbent]['config'])
print('step5')

# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()
# Let's plot the observed losses grouped by budget,
import hpbandster.visualization as hpvis

hpvis.losses_over_time(all_runs)

import matplotlib.pyplot as plt
plt.savefig("random_search.png")

config = id2config[incumbent]['config']
x_train, y_train, x_valid, y_valid, x_test, y_test = mnist.mnist("./")
tmp, tmp, validation_error = mnist.train_and_validate(x_train, y_train, x_valid, y_valid, x_test, y_test, 6, config['learning_rate'], config['num_filters'], config['batch_size'], config['filtersize'])
print("Error: " + str(validation_error))

# TODO: retrain the best configuration (called incumbent) and compute the test error
