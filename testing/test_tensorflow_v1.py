import numpy as np
import tensorflow as tf
from neuraxle.base import ExecutionContext
from neuraxle.hyperparams.distributions import LogUniform
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep

N_SAMPLES = 17

MATMUL_VARIABLE_SCOPE = "matmul"


def create_graph(step: TensorflowV1ModelStep, context: ExecutionContext):
    tf.placeholder('float', name='data_inputs')
    tf.placeholder('float', name='expected_outputs')

    tf.Variable(np.random.rand(), name='weight')
    tf.Variable(np.random.rand(), name='bias')

    return tf.add(tf.multiply(step['data_inputs'], step['weight']), step['bias'])


def create_loss(step: TensorflowV1ModelStep, context: ExecutionContext):
    return tf.reduce_sum(tf.pow(step['output'] - step['expected_outputs'], 2)) / (2 * N_SAMPLES)


def create_optimizer(step: TensorflowV1ModelStep, context: ExecutionContext):
    return tf.train.GradientDescentOptimizer(step.hyperparams['learning_rate'])


def test_tensorflowv1_saver(tmpdir):
    data_inputs = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                            7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    expected_ouptuts = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                                 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    model = Pipeline([create_model_step()])

    for i in range(50):
        model, outputs = model.fit_transform(data_inputs, expected_ouptuts)

    model.save(ExecutionContext(root=tmpdir))

    model = Pipeline([create_model_step()]).load(ExecutionContext(root=tmpdir))
    model, outputs = model.fit_transform(data_inputs, expected_ouptuts)
    assert ((outputs - expected_ouptuts) ** 2).mean() < 0.25


def create_model_step():
    return TensorflowV1ModelStep(
        create_graph=create_graph,
        create_loss=create_loss,
        create_optimizer=create_optimizer,
        has_expected_outputs=True
    ).set_hyperparams(HyperparameterSamples({
        'learning_rate': 0.01
    })).set_hyperparams_space(HyperparameterSpace({
        'learning_rate': LogUniform(0.0001, 0.01)
    }))
