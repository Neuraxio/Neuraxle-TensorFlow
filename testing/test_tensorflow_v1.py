import tensorflow as tf
from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.pipeline import Pipeline

from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1StepSaver, TensorflowV1ModelWrapperMixin

MATMUL_VARIABLE_SCOPE = "matmul"


class TensorflowMatMulModel(TensorflowV1ModelWrapperMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self, savers=[TensorflowV1StepSaver()])

        self.graph = None
        self.sess = None

    def setup(self) -> BaseStep:
        if self.is_initialized:
            return self

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.in_a = tf.placeholder(dtype=tf.float32, shape=2)
            self.in_b = tf.placeholder(dtype=tf.float32, shape=2)

            self.out_a = forward(self.in_a)
            self.out_b = forward(self.in_b)
            self.reg_loss = tf.losses.get_regularization_loss(scope=MATMUL_VARIABLE_SCOPE)

            with tf.variable_scope(MATMUL_VARIABLE_SCOPE, reuse=tf.AUTO_REUSE):
                self.create_session()
                self.is_initialized = True

        return self

    def strip(self):
        self.sess = None
        self.graph = None
        self.in_a = None
        self.in_b = None
        self.out_a = None
        self.out_b = None
        self.reg_loss = None

    def create_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=self.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_session(self):
        return self.sess

    def get_graph(self):
        return self.graph

    def teardown(self):
        if self.sess is not None:
            self.sess.close()
        self.is_initialized = False

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        results = self.sess.run([self.out_a, self.out_b, self.reg_loss],
                                feed_dict={self.in_a: data_inputs[0], self.in_b: data_inputs[1]})
        loss = results[2]
        return self

    def transform(self, data_inputs):
        return self.sess.run([self.out_a, self.out_b],
                             feed_dict={self.in_a: data_inputs[0], self.in_b: data_inputs[1]})


def forward(x):
    with tf.variable_scope(MATMUL_VARIABLE_SCOPE, reuse=tf.AUTO_REUSE):
        W = tf.get_variable("W", initializer=tf.ones(shape=(2, 2)),
                            regularizer=tf.contrib.layers.l2_regularizer(0.04))
        b = tf.get_variable("b", initializer=tf.zeros(shape=2))
        return W * x + b


def test_tensorflowv1_saver(tmpdir):
    model = Pipeline([TensorflowMatMulModel()])

    model = model.fit([[2, 1], [2, 4]])
    model.save(ExecutionContext(root=tmpdir))

    loaded = Pipeline([TensorflowMatMulModel()]).load(ExecutionContext(root=tmpdir))
    outputs = loaded.transform([[3, 1], [5, 4]])

    assert outputs
