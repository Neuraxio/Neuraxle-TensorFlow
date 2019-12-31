import os

import tensorflow as tf
from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v2 import BaseTensorflowV2ModelStep


class LinearModel(tf.keras.Model):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


class Tensorflow2Model(BaseTensorflowV2ModelStep):
    def __init__(self=None, checkpoint_folder=None, hyperparams=None):
        BaseTensorflowV2ModelStep.__init__(self, checkpoint_folder=checkpoint_folder, hyperparams=hyperparams)

    def create_optimizer(self):
        return tf.keras.optimizers.Adam(0.1)

    def create_model(self):
        return LinearModel()

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        x = tf.convert_to_tensor(data_inputs)
        y = tf.convert_to_tensor(expected_outputs)

        with tf.GradientTape() as tape:
            output = self.model(x)
            self.loss = tf.reduce_mean(tf.abs(output - y))

        self.optimizer.apply_gradients(zip(
            tape.gradient(self.loss, self.model.trainable_variables),
            self.model.trainable_variables
        ))

        return self

    def transform(self, data_inputs):
        return self.model(tf.convert_to_tensor(data_inputs))


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat(10).batch(2)


def test_tensorflowv2_saver(tmpdir):
    model = Pipeline([
        Tensorflow2Model(os.path.join(tmpdir, 'tf_checkpoints'))
    ])
    dataset = toy_dataset()
    loss_first_fit = evaluate_model_on_dataset(model, dataset)

    model.save(ExecutionContext(root=tmpdir))

    loaded = Pipeline([
        Tensorflow2Model(os.path.join(tmpdir, 'tf_checkpoints'))
    ]).load(ExecutionContext(root=tmpdir))
    loss_second_fit = evaluate_model_on_dataset(loaded, dataset)
    assert loss_second_fit < (loss_first_fit / 2)


def evaluate_model_on_dataset(model, dataset):
    loss = []
    for example in dataset:
        model, outputs = model.fit_transform(example['x'].numpy(), example['y'].numpy())
        loss.append(model['Tensorflow2Model'].loss.numpy())
    return sum(loss)
