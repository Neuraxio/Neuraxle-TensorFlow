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
from neuraxle.base import BaseSaver, BaseStep, ExecutionContext


class BaseTensorflowV1ModelStep(BaseStep):
    """
    Base class for tensorflow 1 steps.
    It uses :class:`TensorflowV1StepSaver` for saving the model.

    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/checkpoint>`_,
        :class:`~neuraxle.base.BaseStep`
    """
    def __init__(
            self,
            variable_scope=None,
            hyperparams=None
    ):
        BaseStep.__init__(
            self,
            savers=[TensorflowV1StepSaver()],
            hyperparams=hyperparams
        )

        self.variable_scope = variable_scope
        self.tensorflow_props = {}

    def setup(self) -> BaseStep:
        """
        Setup tensorflow 1 graph, and session using a variable scope.

        :return: self
        :rtype: BaseStep
        """
        if self.is_initialized:
            return self

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
                self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=self.graph)

                self.tensorflow_props = self.setup_graph()
                self.tensorflow_props = self.tensorflow_props if self.tensorflow_props is not None else {}

                init = tf.global_variables_initializer()
                self.session.run(init)
                self.is_initialized = True

    def teardown(self) -> BaseStep:
        """
        Close session on teardown.

        :return:
        """
        if self.session is not None:
            self.session.close()
        self.is_initialized = False

        return self

    def strip(self):
        """
        Strip tensorflow 1 properties from to step to make the step serializable.

        :return: stripped step
        :rtype: BaseStep
        """
        self.tensorflow_props = {}
        self.graph = None
        self.session = None

        return self

    @abstractmethod
    def setup_graph(self):
        raise NotImplementedError()

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            return self.fit_model(data_inputs, expected_outputs)

    @abstractmethod
    def fit_model(self, data_inputs, expected_outputs=None) -> BaseStep:
        """
        Fit tensorflow model using the variable scope.

        :param data_inputs: data inputs
        :param expected_outputs: expected outputs to fit on
        :return: fitted self
        :rtype: BaseStep
        """
        raise NotImplementedError()

    def transform(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            return self.transform_model(data_inputs)

    @abstractmethod
    def transform_model(self, data_inputs):
        """
        Transform tensorflow model using the variable scope.

        :param data_inputs:
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, item):
        """
        Get a graph tensor by name using get item.

        :param item: tensor name
        :type item: str

        :return: tensor
        :rtype: tf.Tensor
        """
        if item in self.tensorflow_props:
            return self.tensorflow_props[item]

        return self.graph.get_tensor_by_name(
            "{0}/{1}:0".format(self.variable_scope, item)
        )


class TensorflowV1StepSaver(BaseSaver):
    """
    Step saver for a tensorflow Session using tf.train.Saver().
    It saves, or restores the tf.Session() checkpoint at the context path using the step name as file name.

    .. seealso::
        `Using the saved model format <https://www.tensorflow.org/guide/saved_model>`_,
        :class:`~neuraxle.base.BaseSaver`
    """

    def save_step(self, step: 'BaseTensorflowV1ModelStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Save a step that is using tf.train.Saver().
        :param step: step to save
        :type step: BaseStep
        :param context: execution context to save from
        :type context: ExecutionContext
        :return: saved step
        """
        with step.graph.as_default():
            saver = tf.train.Saver()
            saver.save(step.session, self._get_saved_model_path(context, step))
            step.strip()

        return step

    def load_step(self, step: 'BaseTensorflowV1ModelStep', context: 'ExecutionContext') -> 'BaseStep':
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

        with step.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(step.session, self._get_saved_model_path(context, step))

        return step

    def can_load(self, step: 'BaseTensorflowV1ModelStep', context: 'ExecutionContext'):
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

    def _get_saved_model_path(self, context: ExecutionContext, step: BaseStep):
        """
        Returns the saved model path using the given execution context, and step name.
        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        return os.path.join(context.get_path(), "{0}.ckpt".format(step.get_name()))
