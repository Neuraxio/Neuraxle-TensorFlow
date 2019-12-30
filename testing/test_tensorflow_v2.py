import os

import tensorflow as tf
from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v2 import TensorflowV2ModelWrapperMixin, TensorflowV2StepSaver


class LinearModel(tf.keras.Model):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


class Tensorflow2Model(TensorflowV2ModelWrapperMixin, BaseStep):
    def __init__(self, tensorflow_checkpoint_folder=None):
        BaseStep.__init__(self, savers=[TensorflowV2StepSaver()])
        if tensorflow_checkpoint_folder is None:
            tensorflow_checkpoint_folder = 'tf_chkpts'
        self.tensorflow_checkpoint_folder = tensorflow_checkpoint_folder

    def setup(self) -> BaseStep:
        if self.is_initialized:
            return self

        self.optimizer = tf.keras.optimizers.Adam(0.1)
        self.model = LinearModel()
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.tensorflow_checkpoint_folder,
                                                             max_to_keep=3)
        self.is_initialized = True

        return self

    def get_checkpoint(self):
        return self.checkpoint

    def get_checkpoint_manager(self):
        return self.checkpoint_manager

    def strip(self):
        self.optimizer = None
        self.model = None
        self.checkpoint = None
        self.checkpoint_manager = None
        self.loss = None

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        x = tf.convert_to_tensor(data_inputs)
        y = tf.convert_to_tensor(expected_outputs)
        with tf.GradientTape() as tape:
            output = self.model(x)
            self.loss = tf.reduce_mean(tf.abs(output - y))

        variables = self.model.trainable_variables
        gradients = tape.gradient(self.loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return self

    def transform(self, data_inputs):
        x = tf.convert_to_tensor(data_inputs)
        return self.model(x)


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat(10).batch(2)


def test_tensorflowv2_saver(tmpdir):
    model = Pipeline([
        Tensorflow2Model(tensorflow_checkpoint_folder=os.path.join(tmpdir, 'tf_checkpoints'))
    ])
    dataset = toy_dataset()
    loss_first_fit = evaluate_model_on_dataset(model, dataset)

    model.save(ExecutionContext(root=tmpdir))

    loaded = Pipeline([
        Tensorflow2Model(tensorflow_checkpoint_folder=os.path.join(tmpdir, 'tf_checkpoints'))
    ]).load(ExecutionContext(root=tmpdir))
    loss_second_fit = evaluate_model_on_dataset(loaded, dataset)
    assert loss_second_fit < (loss_first_fit / 2)


def evaluate_model_on_dataset(model, dataset):
    loss = []
    for example in dataset:
        model, outputs = model.fit_transform(example['x'].numpy(), example['y'].numpy())
        loss.append(model['Tensorflow2Model'].loss.numpy())
    return sum(loss)
