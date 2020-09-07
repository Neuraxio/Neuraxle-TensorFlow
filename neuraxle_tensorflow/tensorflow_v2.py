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
import tensorflow as tf
from neuraxle.base import BaseSaver, BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace

from neuraxle_tensorflow.tensorflow import BaseTensorflowModelStep


class Tensorflow2ModelStep(BaseTensorflowModelStep):
    """
    Base class for tensorflow 2 steps.
    It uses :class:`TensorflowV2StepSaver` for saving the model.

    .. seealso::
        `Using the checkpoint model format <https://www.tensorflow.org/guide/checkpoint>`_,
        :class:`~neuraxle.base.BaseStep`
    """
    HYPERPARAMS = HyperparameterSamples({})
    HYPERPARAMS_SPACE = HyperparameterSpace({})

    def __init__(
            self,
            create_model,
            create_loss,
            create_optimizer,
            create_inputs=None,
            data_inputs_dtype=None,
            expected_outputs_dtype=None,
            tf_model_checkpoint_folder=None,
            print_loss=False,
            print_func=None,
            device_name=None
    ):
        BaseTensorflowModelStep.__init__(
            self,
            create_model=create_model,
            create_loss=create_loss,
            create_optimizer=create_optimizer,
            create_inputs=create_inputs,
            data_inputs_dtype=data_inputs_dtype,
            expected_outputs_dtype=expected_outputs_dtype,
            step_saver=TensorflowV2StepSaver(),
            print_loss=print_loss,
            print_func=print_func
        )

        if device_name is None:
            device_name = '/CPU:0'
        self.device_name = device_name

        if tf_model_checkpoint_folder is None:
            tf_model_checkpoint_folder = 'tensorflow_ckpts'
        self.tf_model_checkpoint_folder = tf_model_checkpoint_folder

    def setup(self, context: ExecutionContext) -> BaseStep:
        """
        Setup optimizer, model, and checkpoints for saving.

        :return: step
        :rtype: BaseStep
        """
        if self.is_initialized:
            return self

        with tf.device(self.device_name):
            self.optimizer = self.create_optimizer(self, context)
            self.model = self.create_model(self, context)

            self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint,
                self.tf_model_checkpoint_folder,
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

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        with tf.device(self.device_name):
            self._fit_model(data_inputs, expected_outputs)

        return self

    def _fit_model(self, data_inputs, expected_outputs):
        inputs = self._create_inputs(data_inputs, expected_outputs)
        with tf.GradientTape() as tape:
            output = self.model(inputs, training=True)
            loss = self.create_loss(
                self,
                expected_outputs=tf.convert_to_tensor(expected_outputs, dtype=self.expected_outputs_dtype),
                predicted_outputs=output
            )
            self.add_new_loss(loss)
            self.model.losses.append(loss)

        self.optimizer.apply_gradients(zip(
            tape.gradient(loss, self.model.trainable_variables),
            self.model.trainable_variables
        ))

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        data_inputs = data_container.data_inputs
        expected_outputs = data_container.expected_outputs

        with tf.device(self.device_name):
            output = self._transform_model(data_inputs, expected_outputs)

        data_container.set_data_inputs(output.numpy())
        return data_container

    def _transform_model(self, data_inputs, expected_outputs):
        output = self.model(self._create_inputs(data_inputs), training=False)

        if expected_outputs is not None:
            loss = self.create_loss(
                self,
                expected_outputs=tf.convert_to_tensor(expected_outputs, dtype=self.expected_outputs_dtype),
                predicted_outputs=output
            )
            self.add_new_loss(loss, test_only=True)
        return output

    def transform(self, data_inputs):
        with tf.device(self.device_name):
            output = self.model(self._create_inputs(data_inputs), training=False)
        return output.numpy()

    def _create_inputs(self, data_inputs, expected_outputs=None):
        if self.create_inputs is not None:
            inputs = self.create_inputs(self, data_inputs, expected_outputs)
        else:
            inputs = tf.convert_to_tensor(data_inputs, self.data_inputs_dtype)
        return inputs


class TensorflowV2StepSaver(BaseSaver):
    """
    Step saver for a tensorflow Session using tf.train.Checkpoint().
    It saves, or restores the tf.Session() checkpoint at the context path using the step name as file name.

    .. seealso::
        `Using the checkpoint model format <https://www.tensorflow.org/guide/checkpoint>`_
        :class:`~neuraxle.base.BaseSaver`
    """

    def save_step(self, step: 'Tensorflow2ModelStep', context: 'ExecutionContext') -> 'BaseStep':
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

    def load_step(self, step: 'Tensorflow2ModelStep', context: 'ExecutionContext') -> 'BaseStep':
        """
        Load a step that is using tensorflow using tf.train.Checkpoint().

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        step.is_initialized = False
        step.setup(context)
        step.checkpoint.restore(step.checkpoint_manager.latest_checkpoint)
        return step

    def can_load(self, step: 'Tensorflow2ModelStep', context: 'ExecutionContext') -> bool:
        """
        Returns whether or not we can load.

        :param step: step to load
        :type step: BaseStep
        :param context: execution context to load from
        :type context: ExecutionContext
        :return: loaded step
        """
        return True
