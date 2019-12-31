"""
Neuraxle Tensorflow V2 Utility classes
=========================================
Neuraxle utility classes for tensorflow v2.

..
    Copyright 2019, Neuraxio Inc.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
from abc import abstractmethod

from neuraxle.base import BaseSaver, BaseStep, ExecutionContext
import tensorflow as tf
from tensorflow_core.python.keras.optimizer_v2 import optimizer_v2


class BaseTensorflowV2ModelStep(BaseStep):
    """
    Base class for tensorflow 2 steps.
    It uses :class:`TensorflowV2StepSaver` for saving the model.

    .. seealso::
        `Using the checkpoint model format <https://www.tensorflow.org/guide/checkpoint>`_,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, checkpoint_folder=None, hyperparams=None):
        BaseStep.__init__(
            self,
            savers=[TensorflowV2StepSaver()],
            hyperparams=hyperparams
        )

        if checkpoint_folder is None:
            checkpoint_folder = 'tensorflow_ckpts'
        self.checkpoint_folder = checkpoint_folder

    def setup(self) -> BaseStep:
        """
        Setup optimizer, model, and checkpoints for saving.

        :return: step
        :rtype: BaseStep
        """
        if self.is_initialized:
            return self

        self.optimizer = self.create_optimizer()
        self.model = self.create_model()

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            self.checkpoint_folder,
            max_to_keep=3
        )

        self.is_initialized = True

        return self

    def strip(self):
        """
        Strip tensorflow 2 properties from to step to make it serializable.

        :return:
        """
        self.optimizer = None
        self.model = None
        self.checkpoint = None
        self.checkpoint_manager = None

    @abstractmethod
    def create_optimizer(self) -> optimizer_v2.OptimizerV2:
        """
        Create the tensorflow 2 optimizer to apply gradients.

        :return: tensorflow optimizer v2
        :rtype: optimizer_v2.OptimizerV2
        """
        raise NotImplementedError()

    @abstractmethod
    def create_model(self) -> tf.keras.Model:
        """
        Create the Tensorflow 2 Model to apply gradients.

        :return:
        """
        raise NotImplementedError()


class TensorflowV2StepSaver(BaseSaver):
    """
    Step saver for a tensorflow Session using tf.train.Checkpoint().
    It saves, or restores the tf.Session() checkpoint at the context path using the step name as file name.

    .. seealso::
        `Using the checkpoint model format <https://www.tensorflow.org/guide/checkpoint>`_
        :class:`~neuraxle.base.BaseSaver`
    """

    def save_step(self, step: 'BaseTensorflowV2ModelStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save a step that is using tf.train.Saver().

        :param step: step to save
        :type step: BaseStep
        :param context: execution context to save from
        :type context: ExecutionContext
        :return: saved step
        """
        step.checkpoint_manager.save()
        step.strip()
        return step

    def load_step(self, step: 'BaseTensorflowV2ModelStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load a step that is using tensorflow using tf.train.Checkpoint().

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        step.is_initialized = False
        step.setup()
        step.checkpoint.restore(step.checkpoint_manager.latest_checkpoint)
        return step

    def can_load(self, step: 'BaseTensorflowV2ModelStep', context: 'ExecutionContext') -> bool:
        """
        Returns whether or not we can load.

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        return True
