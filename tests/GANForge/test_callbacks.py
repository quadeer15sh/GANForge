import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from GANForge.callbacks import DCGANVisualization, ConditionalGANVisualization

from GANForge.dcgan import DCGAN
from GANForge.conditional_dcgan import ConditionalDCGAN


class TestModel(Model):

    def __init__(self):
        super().__init__()
        self.linear = Sequential([
            Dense(units=1, input_shape=(1,))
        ])

    def call(self, inputs):
        return self.linear(inputs)


def test_callback_error1():
    model = TestModel()
    model.compile(loss='mse', optimizer='adam')
    x = np.random.randn(1, 1)
    y = np.random.randn(1, 1)

    dcgan_callback = DCGANVisualization(n_epochs=1)

    with pytest.raises(Exception) as e_info:
        model.fit(x, y, epochs=1, callbacks=[dcgan_callback])

    assert f"Invoked for model TestModel ! This callback is available only for DCGAN models" in str(e_info)


def test_callback_error2():
    model = ConditionalDCGAN(input_shape=(32, 32, 1), latent_dim=64, num_classes=3)
    model.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss_fn=tf.keras.losses.BinaryCrossentropy())
    a = np.random.randn(2, 32, 32, 1)
    a[a < -1] = -1
    a[a > 1] = 1
    b = np.random.randint(0, 3, 2)

    dcgan_callback = DCGANVisualization(n_epochs=1)

    with pytest.raises(Exception) as e_info:
        model.fit(a, b, epochs=1, callbacks=[dcgan_callback])

    assert f"Invoked for model ConditionalDCGAN ! This callback is available only for DCGAN models" in str(e_info)


def test_callback_error3():
    model = DCGAN(input_shape=(32, 32, 1), latent_dim=64)
    model.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss_fn=tf.keras.losses.BinaryCrossentropy())
    a = np.random.randn(2, 32, 32, 1)
    a[a < -1] = -1
    a[a > 1] = 1

    cdcgan_callback = ConditionalGANVisualization(n_epochs=1, labels=['cat', 'dog', 'bear'])

    with pytest.raises(Exception) as e_info:
        model.fit(a, epochs=1, callbacks=[cdcgan_callback])

    assert f"Invoked for model DCGAN ! This callback is available only for Conditional DCGAN models" in str(e_info)
