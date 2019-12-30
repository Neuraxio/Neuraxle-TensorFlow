"""
Neuraxle Tensorflow V1 Utility classes
=========================================
Neuraxle utility classes for tensorflow v1.

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
import os
from abc import abstractmethod

import tensorflow as tf

from neuraxle.base import BaseSaver


class TensorflowV1ModelWrapperMixin:
    """
    A class that represents a step that contains a tensorflow v1 model.

    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/saved_model>`_
    """
    @abstractmethod
    def get_session(self):
        pass

    @abstractmethod
    def get_graph(self):
        pass

    @abstractmethod
    def strip(self):
        """
        Get the tensorflow tf.Graph() object.

        :return: tf.Graph
        """
        raise NotImplementedError()


class TensorflowV1StepSaver(BaseSaver):
    """
    Step saver for a tensorflow Session using tf.train.Saver().
    It saves, or restores the tf.Session() checkpoint at the context path using the step name as file name.
    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/saved_model>`_
    """

    def save_step(self, step: 'TensorflowV1ModelWrapperMixin', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save a step that is using tf.train.Saver().
        :param step: step to save
        :type step: BaseStep
        :param context: execution context to save from
        :type context: ExecutionContext
        :return: saved step
        """
        with step.get_graph().as_default():
            saver = tf.train.Saver()
            saver.save(
                step.get_session(),
                self._get_saved_model_path(context, step)
            )

            step.strip()

        return step

    def load_step(self, step: 'TensorflowV1ModelWrapperMixin', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load a step that is using tensorflow using tf.train.Saver().
        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        step.is_initialized = False
        step.setup()
        with step.get_graph().as_default():
            saver = tf.train.Saver()
            saver.restore(
                step.get_session(),
                self._get_saved_model_path(context, step)
            )

        return step

    def can_load(self, step: 'TensorflowV1ModelWrapperMixin', context: 'ExecutionContext'):
        """
        Returns whether or not we can load.
        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        meta_exists = os.path.exists(os.path.join(context.get_path(), "{0}.ckpt.meta".format(step.get_name())))
        index_exists = os.path.exists(os.path.join(context.get_path(), "{0}.ckpt.index".format(step.get_name())))
        return meta_exists and index_exists

    def _get_saved_model_path(self, context, step):
        """
        Returns the saved model path using the given execution context, and step name.
        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        return os.path.join(
            context.get_path(),
            "{0}.ckpt".format(step.get_name())
        )
