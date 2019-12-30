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
                'learning_rate': 0.01
            })
        )

    def setup_graph(self) -> Dict:
        forward(self.hyperparams)
        tf.equal(tf.argmax(self['pred'], 1), tf.argmax(self['y'], 1), name='correct_pred')
        tf.reduce_mean(tf.cast(self['correct_pred'], tf.float32), name='accuracy')

        l2 = self.hyperparams['lambda_loss_amount'] * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self['y'], logits=self['pred'])) + l2

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyperparams['learning_rate']
        ).minimize(loss)

        return {
            'loss': loss,
            'optimizer': optimizer
        }

    def fit_model(self, data_inputs, expected_outputs=None) -> BaseStep:
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        if not isinstance(expected_outputs, np.ndarray):
            expected_outputs = np.array(expected_outputs)

        if expected_outputs.shape != (len(data_inputs), self.hyperparams['n_classes']):
            expected_outputs = np.reshape(expected_outputs, (len(data_inputs), self.hyperparams['n_classes']))

            _, loss, acc = self.session.run(
                [self['optimizer'], self['loss'], self['accuracy']],
                feed_dict={
                    self['x']: data_inputs,
                    self['y']: expected_outputs
                }
            )

            print("Batch Loss = " + "{:.6f}".format(loss) + ", Accuracy = {}".format(acc))

        self.is_invalidated = True

        return self

    def transform_model(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return self.session.run(
            [self['pred']],
            feed_dict={self['x']: data_inputs}
        )[0]


def forward(hyperparams):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, hyperparams['n_steps'], hyperparams['n_inputs']], name='x')
    y = tf.placeholder(tf.float32, [None, hyperparams['n_classes']], name='y')

    # Graph weights
    weights = {
        'hidden': tf.Variable(
            tf.random_normal([hyperparams['n_inputs'], hyperparams['n_hidden']])
        ),  # Hidden layer weights
        'out': tf.Variable(
            tf.random_normal([hyperparams['n_hidden'], hyperparams['n_classes']], mean=1.0)
        )
    }

    biases = {
        'hidden': tf.Variable(
            tf.random_normal([hyperparams['n_hidden']])
        ),
        'out': tf.Variable(
            tf.random_normal([hyperparams['n_classes']])
        )
    }

    data_inputs = tf.transpose(
        x,
        [1, 0, 2])  # permute n_steps and batch_size

    # Reshape to prepare input to hidden activation
    data_inputs = tf.reshape(data_inputs, [-1, hyperparams['n_inputs']])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(
        tf.matmul(data_inputs, weights['hidden']) + biases['hidden']
    )

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, hyperparams['n_steps'], 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(hyperparams['n_hidden'], forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(hyperparams['n_hidden'], forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    pred = tf.matmul(lstm_last_output, weights['out']) + biases['out']
    return tf.identity(pred, name='pred')


def test_tensorflowv1_saver(tmpdir):
    model = Pipeline([
        Tensorflow1Model()
    ])

    model.save(ExecutionContext(root=tmpdir))

    loaded = Pipeline([Tensorflow1Model()]).load(ExecutionContext(root=tmpdir))
    outputs = loaded.transform([[3, 1], [5, 4]])

    assert outputs
