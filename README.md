# Neuraxle-TensorFlow

TensorFlow steps, savers, and utilities for [Neuraxle](https://github.com/Neuraxio/Neuraxle).

Neuraxle is a Machine Learning (ML) library for building neat pipelines, providing the right abstractions to both ease research, development, and deployment of your ML applications.

## Usage example

[See also a complete example](https://github.com/Neuraxio/LSTM-Human-Activity-Recognition/blob/neuraxle-refactor/steps/lstm_rnn_tensorflow_model_wrapper.py)

### Tensorflow 1

Create a tensorflow 1 model step by giving it a graph, an optimizer, and a loss function. 

```python
def create_graph(step: TensorflowV1ModelStep):
    tf.placeholder('float', name='data_inputs')
    tf.placeholder('float', name='expected_outputs')

    tf.Variable(np.random.rand(), name='weight')
    tf.Variable(np.random.rand(), name='bias')
    
    return tf.add(tf.multiply(step['data_inputs'], step['weight']), step['bias'])
    
"""
# Note: you can also return a tuple containing two elements : tensor for training (fit), tensor for inference (transform)
def create_graph(step: TensorflowV1ModelStep)
    # ...
    decoder_outputs_training = create_training_decoder(step, encoder_state, decoder_cell)
    decoder_outputs_inference = create_inference_decoder(step, encoder_state, decoder_cell)

    return decoder_outputs_training, decoder_outputs_inference
"""


def create_loss(step: TensorflowV1ModelStep):
    return tf.reduce_sum(tf.pow(step['output'] - step['expected_outputs'], 2)) / (2 * N_SAMPLES)

def create_optimizer(step: TensorflowV1ModelStep):
    return tf.train.GradientDescentOptimizer(step.hyperparams['learning_rate'])

model_step = TensorflowV1ModelStep(
    create_grah=create_graph,
    create_loss=create_loss,
    create_optimizer=create_optimizer,
    has_expected_outputs=True
).set_hyperparams(HyperparameterSamples({
    'learning_rate': 0.01
})).set_hyperparams_space(HyperparameterSpace({
    'learning_rate': LogUniform(0.0001, 0.01)
}))
```

### Tensorflow 2

Create a tensorflow 2 model step by giving it a model, an optimizer, and a loss function. 

```python
def create_model(step: Tensorflow2ModelStep):
    return LinearModel()

def create_optimizer(step: Tensorflow2ModelStep):
    return tf.keras.optimizers.Adam(0.1)

def create_loss(step: Tensorflow2ModelStep, expected_outputs, predicted_outputs):
    return tf.reduce_mean(tf.abs(predicted_outputs - expected_outputs))

model_step = Tensorflow2ModelStep(
    create_model=create_model,
    create_optimizer=create_optimizer,
    create_loss=create_loss,
    tf_model_checkpoint_folder=os.path.join(tmpdir, 'tf_checkpoints')
)
```
