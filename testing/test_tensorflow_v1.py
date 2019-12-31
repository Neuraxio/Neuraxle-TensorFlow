from typing import Dict

import numpy as np
import tensorflow as tf
from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v1 import BaseTensorflowV1ModelStep

N_SAMPLES = 17

MATMUL_VARIABLE_SCOPE = "matmul"


class Tensorflow1Model(BaseTensorflowV1ModelStep):
    def __init__(self, variable_scope=None):
        if variable_scope is None:
            variable_scope = 'Tensorflow1Model'
        BaseTensorflowV1ModelStep.__init__(
            self,
            variable_scope=variable_scope,
            hyperparams=HyperparameterSamples({
                'learning_rate': 0.01
            })
        )

    def setup_graph(self) -> Dict:
        tf.placeholder('float', name='x')
        tf.placeholder('float', name='y')

        tf.Variable(np.random.rand(), name='weight')
        tf.Variable(np.random.rand(), name='bias')

        tf.add(tf.multiply(self['x'], self['weight']), self['bias'], name='pred')

        loss = tf.reduce_sum(tf.pow(self['pred'] - self['y'], 2)) / (2 * N_SAMPLES)
        optimizer = tf.train.GradientDescentOptimizer(self.hyperparams['learning_rate']).minimize(loss)

        return {
            'loss': loss,
            'optimizer': optimizer
        }

    def fit_model(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        for (x, y) in zip(data_inputs, expected_outputs):
            self.session.run(self['optimizer'], feed_dict={self['x']: x, self['y']: y})

        self.is_invalidated = True

        return self

    def transform_model(self, data_inputs):
        return self.session.run(self['weight']) * data_inputs + self.session.run(self['bias'])


def test_tensorflowv1_saver(tmpdir):
    data_inputs = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                            7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    expected_ouptuts = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                                 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    model = Pipeline([Tensorflow1Model()])
    for i in range(50):
        model, outputs = model.fit_transform(data_inputs, expected_ouptuts)

    model.save(ExecutionContext(root=tmpdir))

    model = Pipeline([Tensorflow1Model()]).load(ExecutionContext(root=tmpdir))
    model, outputs = model.fit_transform(data_inputs, expected_ouptuts)
    assert ((outputs - expected_ouptuts) ** 2).mean() < 0.25
