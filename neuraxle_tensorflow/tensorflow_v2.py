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

from neuraxle.base import BaseSaver


class TensorflowV2ModelWrapperMixin:
    """
    A class that represents a step that contains a tensorflow v2 model.

    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/checkpoint>`_
    """

    @abstractmethod
    def get_checkpoint(self):
        pass

    @abstractmethod
    def get_checkpoint_manager(self):
        pass

    @abstractmethod
    def strip(self):
        """
        Get the tensorflow tf.Graph() object.

        :return: tf.Graph
        """
        raise NotImplementedError()


class TensorflowV2StepSaver(BaseSaver):
    """
    Step saver for a tensorflow Session using tf.train.Saver().
    It saves, or restores the tf.Session() checkpoint at the context path using the step name as file name.
    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/saved_model>`_
    """

    def save_step(self, step: 'TensorflowV2ModelWrapperMixin', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save a step that is using tf.train.Saver().
        :param step: step to save
        :type step: BaseStep
        :param context: execution context to save from
        :type context: ExecutionContext
        :return: saved step
        """
        step.get_checkpoint_manager().save()
        step.strip()
        return step

    def load_step(self, step: 'TensorflowV2ModelWrapperMixin', context: 'ExecutionContext') -> 'BaseStep':
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
        step.get_checkpoint().restore(step.get_checkpoint_manager().latest_checkpoint)
        return step

    def can_load(self, step: 'TensorflowV2ModelWrapperMixin', context: 'ExecutionContext') -> bool:
        """
        Returns whether or not we can load.

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        return True
