import os

import tensorflow as tf
from neuraxle.base import ExecutionContext
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v2 import Tensorflow2ModelStep


class LinearModel(tf.keras.Model):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat(10).batch(2)


def create_model(step: Tensorflow2ModelStep, context: ExecutionContext):
    return LinearModel()


def create_optimizer(step: Tensorflow2ModelStep, context: ExecutionContext):
    return tf.keras.optimizers.Adam(0.1)


def create_loss(step: Tensorflow2ModelStep, expected_outputs, predicted_outputs):
    return tf.reduce_mean(tf.abs(predicted_outputs - expected_outputs))


def test_tensorflowv2_saver(tmpdir):
    dataset = toy_dataset()
    model = Pipeline([
        create_model_step(tmpdir)
    ])
    loss_first_fit = evaluate_model_on_dataset(model, dataset)

    model.save(ExecutionContext(root=tmpdir))

    loaded = Pipeline([
        create_model_step(tmpdir)
    ]).load(ExecutionContext(root=tmpdir))
    loss_second_fit = evaluate_model_on_dataset(loaded, dataset)

    assert loss_second_fit < (loss_first_fit / 2)


def create_model_step(tmpdir):
    return Tensorflow2ModelStep(
        create_model=create_model,
        create_optimizer=create_optimizer,
        create_loss=create_loss,
        tf_model_checkpoint_folder=os.path.join(tmpdir, 'tf_checkpoints')
    )


def evaluate_model_on_dataset(model, dataset):
    loss = []
    for example in dataset:
        expected_outputs = example['y'].numpy()
        model, outputs = model.fit_transform(example['x'].numpy(), expected_outputs)
        loss.append(((outputs - expected_outputs) ** 2).mean())

    return sum(loss)
