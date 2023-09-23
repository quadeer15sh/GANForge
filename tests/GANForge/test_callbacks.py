import pytest
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from GANForge.callbacks import DCGANVisualization


class TestModel(Model):

    def __init__(self):
        super().__init__()
        self.linear = Sequential([
            Dense(units=1, input_shape=(1,))
        ])

    def call(self, inputs):
        return self.linear(inputs)


def test_dcgan_callback_error():
    model = TestModel()
    model.compile(loss='mse', optimizer='adam')
    x = np.random.randn(1, 1)
    y = np.random.randn(1, 1)

    dcgan_callback = DCGANVisualization(n_epochs=1)

    with pytest.raises(Exception) as e_info:
        model.fit(x, y, epochs=1, callbacks=[dcgan_callback])

    assert f"Invoked for model {type(model).__name__} ! This callback is available only for DCGAN models" in str(e_info)
