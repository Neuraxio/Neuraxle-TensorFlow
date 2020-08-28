import os

from neuraxle.metaopt.auto_ml import HyperparamsJSONRepository, AutoML, ValidationSplitter
from neuraxle.metaopt.callbacks import ScoringCallback
from neuraxle.metaopt.trial import TRIAL_STATUS, Trials
from neuraxle.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from neuraxle_tensorflow.observable_classes import Observable
from neuraxle_tensorflow.tensorboard import TensorBoardTrialObserver

class TrialsSequentialObservable(Observable):
    def __init__(self, trials: Trials):
        super().__init__()
        self.trials = trials

    def notify_all_trials_sequentially(self, sleep_between_trials: int = 0, sleep_between_epochs: int = 0):
        # TODO: neurodata implement this (see https://www.neuraxle.org/stable/api/neuraxle.metaopt.trial.html)
        # TODO: neurodata read (https://github.com/Neuraxio/Neuraxle/blob/20c6e57/neuraxle/metaopt/trial.py)
        # TODO: neurodata notify observers with self.on_next(trial)....
        pass

def test_tensorboard_trial_observer_should_write_trial_log_files(tmpdir):
    # TODO: try with your own folder, and tensorboard to see if the plots are being updated in real time
    tensorboard_logging_folder: str = os.path.join(tmpdir, 'tensorboard')
    hyperparams_repository = HyperparamsJSONRepository(
        cache_folder=os.path.join(tmpdir, 'cache'),
        best_retrained_model_folder=os.path.join(tmpdir, 'best')
    )
    # TODO: neurodata create data_inputs, and expected_outputs (see test_tensorflow_v2.py)
    data_inputs = []
    expected_outputs = []

    p = Pipeline([
        # TODO: neurodata create a simple linear regression pipeline that can fit incrementally (see test_tensorflow_v2.py)
    ])

    _fit_automl(p, data_inputs, expected_outputs, hyperparams_repository)

    trials: Trials = hyperparams_repository.load_all_trials(status=TRIAL_STATUS.SUCCESS)
    trials_observable = TrialsSequentialObservable(trials=trials)

    tensorboard_observer: TensorBoardTrialObserver = TensorBoardTrialObserver(logging_folder=tensorboard_logging_folder)
    trials_observable.subscribe(tensorboard_observer)

    trials_observable.notify_all_trials_sequentially(
        sleep_between_trials=0, # TODO: try with other values here to see if the real time updates work (on your own not inside this unit test)
        sleep_between_epochs=0 # TODO: try with other values here to see if the real time updates work (on your own not inside this unit test)
    )
    # TODO: neurodata assert all trial log files have been written

def test_tensorboard_should_plot_loss(tmpdir):
    # TODO: wait for more instructions here...
    # NICE TO HAVE ;)
    # TODO: update the trial inside the execution context
    # TODO: add the model loss in the trial metric results
    pass

def test_tensorboard_should_plot_model_weights(tmpdir):
    # TODO: wait for more instructions here...
    # TODO: read on tensorboard documentation how to plot model weights, and test it here
    pass


def _fit_automl(p, data_inputs, expected_outputs, hyperparams_repository):
    auto_ml: AutoML = AutoML(
        pipeline=p,
        validation_splitter=ValidationSplitter(test_size=0.10),
        scoring_callback=ScoringCallback(mean_squared_error, higher_score_is_better=False),
        hyperparams_repository=hyperparams_repository,
        refit_trial=True
    )

    auto_ml.fit(data_inputs=data_inputs, expected_outputs=expected_outputs)
