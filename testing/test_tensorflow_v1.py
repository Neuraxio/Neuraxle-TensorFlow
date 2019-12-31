from typing import Dict

import numpy as np
import tensorflow as tf
from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v1 import BaseTensorflowV1ModelStep

MATMUL_VARIABLE_SCOPE = "matmul"


class Tensorflow1Model(BaseTensorflowV1ModelStep):
    def __init__(self, variable_scope=None):
        if variable_scope is None:
            variable_scope = 'Tensorflow1Model'
        BaseTensorflowV1ModelStep.__init__(
            self,
            variable_scope=variable_scope,
            hyperparams=HyperparameterSamples({
                'learning_rate': 0.01,
                'lambda_loss_amount': 0.0015
            })
        )

    def setup_graph(self) -> Dict:
        tf.placeholder(tf.float32, [None, None], name='x')
        tf.placeholder(tf.float32, [None, None], name='y')

        l2 = self.hyperparams['lambda_loss_amount'] * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        pred = tf.keras.layers.Dense(5)
        loss = tf.reduce_mean(pred) + l2
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyperparams['learning_rate']
        ).minimize(loss)

        tf.equal(tf.argmax(self['pred'], 1), tf.argmax(self['y'], 1), name='correct_pred')
        tf.reduce_mean(tf.cast(self['correct_pred'], tf.float32), name='accuracy')

        return {
            'pred': pred,
            'loss': loss,
            'optimizer': optimizer
        }

    def fit_model(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        _, loss, acc = self.session.run(
            [self['optimizer'], self['loss'], self['accuracy']],
            feed_dict={
                self['x']: np.array(data_inputs),
                self['y']: np.array(expected_outputs)
            }
        )
        self.loss = loss

        self.is_invalidated = True

        return self

    def transform_model(self, data_inputs):
        return self.session.run(
            [self['pred']],
            feed_dict={self['x']: np.array(data_inputs)}
        )[0]


def toy_dataset():
    return [
        ([[0.], [1.]], [[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]]),
        ([[2.], [3.]], [[10., 11., 12., 13., 14.], [15., 16., 17., 18., 19.]]),
        ([[4.], [5.]], [[20., 21., 22., 23., 24.], [25., 26., 27., 28., 29.]]),
        ([[6.], [7.]], [[30., 31., 32., 33., 34.], [35., 36., 37., 38., 39.]])
    ]


def test_tensorflowv1_saver(tmpdir):
    model = Pipeline([Tensorflow1Model()])
    dataset = toy_dataset()
    loss_first_fit = evaluate_model_on_dataset(model, dataset)

    model.save(ExecutionContext(root=tmpdir))

    loaded = Pipeline([Tensorflow1Model()]).load(ExecutionContext(root=tmpdir))
    loss_second_fit = evaluate_model_on_dataset(loaded, dataset)
    assert loss_second_fit < (loss_first_fit / 2)


def evaluate_model_on_dataset(model, dataset):
    loss = []
    for x, y in dataset:
        model, outputs = model.fit_transform(x, y)
        loss.append(model['Tensorflow1Model'].loss)
    return sum(loss)
