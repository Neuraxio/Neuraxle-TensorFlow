from typing import Tuple

from neuraxle.metaopt.auto_ml import HyperparamsRepository
from neuraxle.metaopt.trial import Trial, TrialSplit
import tensorflow as tf
import os

from tensorflow.python.ops.summary_ops_v2 import ResourceSummaryWriter
from neuraxle_tensorflow.observable_classes import Observer


class TensorBoardTrialObserver(Observer):
    def __init__(self, logging_folder: str):
        super().__init__()
        self.logging_folder: str = logging_folder

    def on_next(self, arg: Tuple[HyperparamsRepository, Trial]):
        hyperparams_repo, trial = arg

        trial_hyperparams: dict = trial.hyperparams.to_flat_as_dict_primitive()
        trial_hash: str = hyperparams_repo._get_trial_hash(trial_hyperparams)

        train_log_dir: str = os.path.join(self.logging_folder, trial_hash, 'train_values')
        validation_log_dir: str = os.path.join(self.logging_folder, trial_hash, 'validation_values')

        train_summary_writer: ResourceSummaryWriter = tf.summary.create_file_writer(train_log_dir)
        validation_summary_writer: ResourceSummaryWriter  = tf.summary.create_file_writer(validation_log_dir)

        for split in trial.validation_splits:
            self._plot_trial_split(
                split=split,
                train_summary_writer=train_summary_writer,
                validation_summary_writer=validation_summary_writer
            )

    def _plot_trial_split(
            self,
            split: TrialSplit,
            train_summary_writer: ResourceSummaryWriter,
            validation_summary_writer: ResourceSummaryWriter
    ):
        for metric_name, values in split.metrics_results.items():
            validation_values = values['validation_values']
            train_values = values['train_values']

            for epoch, (train_value, validation_value) in enumerate(zip(train_values, validation_values)):
                with train_summary_writer.as_default():
                    tf.summary.scalar(metric_name, train_value, step=epoch)

                with validation_summary_writer.as_default():
                    tf.summary.scalar(metric_name, validation_value, step=epoch)
