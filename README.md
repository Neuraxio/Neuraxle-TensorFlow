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
class YourTensorflowModelWrapper(BaseStep):
    def __init__(self):
        BaseStep.__init__(
            self, 
            hyperparams=HYPERPARAMS,
            savers=[TensorflowV1StepSaver()]
        )
```

[See also a complete example](https://github.com/Neuraxio/LSTM-Human-Activity-Recognition/blob/neuraxle-refactor/steps/lstm_rnn_tensorflow_model_wrapper.py)
