# Neuraxle-TensorFlow

TensorFlow steps, savers, and utilities for [Neuraxle](https://github.com/Neuraxio/Neuraxle).

Neuraxle is a Machine Learning (ML) library for building neat pipelines, providing the right abstractions to both ease research, development, and deployment of your ML applications.

## Installation for tensorflow>=1.15

```
neuraxle-tensorflow[tf]
```

## Installation for tensorflow-gpu>=1.15

```
neuraxle-tensorflow[tf_gpu]
```

## Usage example

```python
class YourTensorflowModelWrapper(TensorflowV1ModelWrapperMixin, BaseStep):
    def __init__(self):
        TensorflowV1ModelWrapperMixin.__init__(self)
        BaseStep.__init__(
            self, 
            hyperparams=HYPERPARAMS,
            savers=[TensorflowV1StepSaver()]
        )

    def setup(self) -> BaseStep:
        if self.is_initialized:
            return self

        self.create_graph()

        with self.graph.as_default():
            # Launch the graph
            with tf.variable_scope(LSTM_RNN_VARIABLE_SCOPE, reuse=tf.AUTO_REUSE):
                pred = tf_model_forward(PRED_NAME, X_NAME, Y_NAME, self.hyperparams)

                # Loss, optimizer and evaluation
                # L2 loss prevents this overkill neural network to overfit the data

                l2 = self.hyperparams['lambda_loss_amount'] * sum(
                    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
                )

                # Softmax loss
                self.cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.get_y_placeholder(),
                        logits=pred
                    )
                ) + l2

                # Adam Optimizer
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.hyperparams['learning_rate']
                ).minimize(self.cost)

                self.correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.get_tensor_by_name(Y_NAME), 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

                # To keep track of training's performance
                self.test_losses = []
                self.test_accuracies = []
                self.train_losses = []
                self.train_accuracies = []

                self.create_session()

                self.is_initialized = True

        return self

     def create_graph(self):
        self.graph = tf.Graph()

    def create_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True), graph=self.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def get_tensor_by_name(self, name):
        return self.graph.get_tensor_by_name("{0}/{1}:0".format(LSTM_RNN_VARIABLE_SCOPE, name))

    def get_graph(self):
        return self.graph

    def get_session(self):
        return self.sess
    
    # ....
    
```

[See also a complete example](https://github.com/Neuraxio/LSTM-Human-Activity-Recognition/blob/neuraxle-refactor/steps/lstm_rnn_tensorflow_model_wrapper.py)
