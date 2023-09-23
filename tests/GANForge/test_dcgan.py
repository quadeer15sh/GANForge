import pytest
from GANForge.dcgan import DCGAN
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dropout, Flatten, Dense, Input,
                                     BatchNormalization, Conv2DTranspose,
                                     LeakyReLU, Reshape)


def test_dcgan_success1():
    model = DCGAN(input_shape=(32, 32, 3), latent_dim=100)
    assert model.latent_dim == 100


def test_dcgan_success2():
    input_shape = (32, 32, 1)
    latent_dim = 100
    _discriminator = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(64, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(256, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(256, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    g_h = _discriminator.layers[-7].output_shape[1]
    g_w = _discriminator.layers[-7].output_shape[2]
    g_d = _discriminator.layers[-7].output_shape[3]

    _generator = Sequential(
        [
            Input(shape=(latent_dim,)),
            Dense(g_h * g_w * g_d),
            Reshape((g_h, g_w, g_d)),
            Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(input_shape[2], kernel_size=5, padding="same", activation="tanh"),
        ],
        name="generator",
    )

    model = DCGAN(input_shape=input_shape, latent_dim=latent_dim)
    assert model.generator.input_shape == _generator.input_shape
    assert model.generator.output_shape == _generator.output_shape
    assert model.discriminator.input_shape == _discriminator.input_shape
    assert model.discriminator.output_shape == _discriminator.output_shape


def test_dcgan_success3():
    input_shape = (32, 32, 1)
    latent_dim = 100
    _discriminator = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(64, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(256, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    g_h = _discriminator.layers[-7].output_shape[1]
    g_w = _discriminator.layers[-7].output_shape[2]
    g_d = _discriminator.layers[-7].output_shape[3]

    _generator = Sequential(
        [
            Input(shape=(latent_dim,)),
            Dense(g_h * g_w * g_d),
            Reshape((g_h, g_w, g_d)),
            Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(input_shape[2], kernel_size=5, padding="same", activation="tanh"),
        ],
        name="generator",
    )

    model = DCGAN(discriminator=_discriminator, generator=_generator)
    assert model.generator.input_shape == _generator.input_shape
    assert model.generator.output_shape == _generator.output_shape
    assert model.discriminator.input_shape == _discriminator.input_shape
    assert model.discriminator.output_shape == _discriminator.output_shape


def test_dcgan_success4():
    latent_dim = 128
    random_latent_vectors = tf.random.normal(shape=(2, latent_dim))
    model = DCGAN(input_shape=(32, 32, 1), latent_dim=128)
    output = model(random_latent_vectors)
    assert output.shape == (2, 32, 32, 1)


def test_dcgan_error1():
    discriminator = Sequential(
        [
            Input(shape=(32, 32, 1)),
            Conv2D(32, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2D(64, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Conv2D(128, kernel_size=5, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(0.2),
            Flatten(),
            Dropout(0.4),
            Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
    with pytest.raises(Exception) as e_info:
        DCGAN(discriminator=discriminator)

    assert "Both discriminator and generator should be passed, but only discriminator was found." in str(e_info)


def test_dcgan_error2():
    with pytest.raises(Exception) as e_info:
        DCGAN(latent_dim=128)

    assert "Both input_shape and latent_dim should be passed, but only latent_dim was found." in str(e_info)


def test_dcgan_error3():
    with pytest.raises(Exception) as e_info:
        DCGAN(input_shape=(28, 28, 1))

    assert "Both input_shape and latent_dim should be passed, but only input_shape was found." in str(e_info)


def test_dcgan_error4():
    with pytest.raises(Exception) as e_info:
        DCGAN(input_shape=(28, 28), latent_dim=128)

    assert "The input shape must be provided in (height, width, channels) format" in str(e_info)


def test_dcgan_error5():
    with pytest.raises(Exception) as e_info:
        DCGAN(input_shape=(28, 28, 1), latent_dim=128)

    assert "The input_shape does not match with the output shape of the default generator" in str(e_info)


def test_dcgan_error6():
    model_input = np.random.randn(2, 2)
    model = DCGAN(input_shape=(32, 32, 1), latent_dim=128)
    with pytest.raises(Exception):
        model(model_input)