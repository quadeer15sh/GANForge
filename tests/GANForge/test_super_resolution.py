import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Input,
                                     BatchNormalization, LeakyReLU, PReLU,
                                     GlobalAvgPool2D, Add)
from tensorflow.keras.models import Sequential, Model

from GANForge.losses import PerceptualLoss
from GANForge.super_resolution import SRGAN


@pytest.fixture
def generator_model():
    input_image = Input(shape=(24, 24, 3))
    x = Conv2D(64, kernel_size=9, padding='same')(input_image)
    p_relu_output = PReLU(shared_axes=[1, 2])(x)
    x_in = p_relu_output

    for i in range(2):
        x = Conv2D(64, kernel_size=3, padding='same')(x_in)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(64, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x_in = Add()([x_in, x])

    x = Conv2D(64, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization()(x)
    x = Add()([x, p_relu_output])

    x = Conv2D(256, kernel_size=3, padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(256, kernel_size=3, padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)
    x = PReLU(shared_axes=[1, 2])(x)

    output = Conv2D(3, kernel_size=9, padding='same')(x)

    return Model(inputs=input_image, outputs=output)


@pytest.fixture
def discriminator_model():
    discriminator = Sequential(name='discriminator')
    discriminator.add(Input(shape=(96, 96, 3)))
    discriminator.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    discriminator.add(BatchNormalization())
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(GlobalAvgPool2D())
    discriminator.add(Dense(64))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, activation="sigmoid"))

    return discriminator


def test_srgan_success1():
    model = SRGAN(input_shape_lr=(24, 24, 3),
                  input_shape_hr=(96, 96, 3),
                  num_residual_blocks=4,
                  num_disc_blocks=2)

    assert model.num_residual_blocks == 4
    assert model.num_disc_blocks == 2


def test_srgan_success2():
    model = SRGAN(input_shape_lr=(32, 32, 3),
                  input_shape_hr=(256, 256, 3),
                  scaling_factor=8,
                  num_residual_blocks=2,
                  num_disc_blocks=2)

    assert model.scaling_factor == 8


def test_srgan_success3():
    model = SRGAN(input_shape_lr=(32, 32, 3),
                  input_shape_hr=(256, 256, 3),
                  scaling_factor=8,
                  num_residual_blocks=2,
                  num_disc_blocks=2)

    assert model.scaling_factor == 8


def test_srgan_success4(generator_model, discriminator_model):
    generator = generator_model
    discriminator = discriminator_model

    model = SRGAN(input_shape_lr=(24, 24, 3),
                  input_shape_hr=(96, 96, 3),
                  scaling_factor=4,
                  discriminator=discriminator,
                  generator=generator)

    assert model.generator.input_shape[1:] == generator.input_shape[1:]
    assert model.generator.output_shape[1:] == generator.output_shape[1:]
    assert model.discriminator.layers[0].input_shape == discriminator.layers[0].input_shape
    assert model.discriminator.layers[-1].output_shape == discriminator.layers[-1].output_shape
    assert len(model.discriminator.layers) == len(discriminator.layers)


def test_srgan_success5():
    lr = np.random.randn(2, 12, 12, 3)
    hr = np.random.randn(2, 48, 48, 3)

    model = SRGAN(input_shape_lr=(12, 12, 3),
                  input_shape_hr=(48, 48, 3),
                  scaling_factor=4,
                  num_residual_blocks=1,
                  num_disc_blocks=1)

    model.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  d_loss=tf.keras.losses.BinaryCrossentropy(),
                  g_loss=PerceptualLoss(activation_layer=2, weights=None))

    model.fit(lr, hr, epochs=1)


def test_srgan_success6():
    lr = np.random.randn(2, 12, 12, 3)

    model = SRGAN(input_shape_lr=(12, 12, 3),
                  input_shape_hr=(48, 48, 3),
                  scaling_factor=4,
                  num_residual_blocks=1,
                  num_disc_blocks=1)
    sr_image = model.generator.predict(lr)

    assert sr_image.shape == (2, 48, 48, 3)


def test_srgan_error1():

    with pytest.raises(Exception) as ex:
        SRGAN(input_shape_lr=(32, 32, 3),
              input_shape_hr=(256, 256, 3),
              num_residual_blocks=2,
              num_disc_blocks=2)

    assert (f"Error: The SRGAN model scales low resolution images to high resolution "
            f"by a factor of 4, "
            f"found: 8") in str(ex)


def test_srgan_error2(generator_model, discriminator_model):
    generator = generator_model
    discriminator = discriminator_model

    with pytest.raises(Exception) as ex:
        SRGAN(input_shape_lr=(32, 32, 3),
              input_shape_hr=(128, 128, 3),
              num_residual_blocks=2,
              num_disc_blocks=2,
              generator=generator,
              discriminator=discriminator)

    assert "Error: input_shape_hr does not match the input shape of the discriminator" in str(ex)


def test_srgan_error3(generator_model, discriminator_model):
    generator = generator_model
    discriminator = discriminator_model

    with pytest.raises(Exception) as ex:
        SRGAN(input_shape_lr=(48, 48, 3),
              input_shape_hr=(96, 96, 3),
              scaling_factor=2,
              num_residual_blocks=2,
              num_disc_blocks=2,
              generator=generator,
              discriminator=discriminator)

    assert "Error: input_shape_lr does not match the input shape of the generator" in str(ex)


def test_srgan_error4(generator_model):
    generator = generator_model

    with pytest.raises(Exception) as ex:
        SRGAN(input_shape_lr=(32, 32, 3),
              input_shape_hr=(128, 128, 3),
              num_residual_blocks=2,
              num_disc_blocks=2,
              generator=generator)

    assert "Both discriminator and generator should be passed, but only generator was found." in str(ex)


def test_srgan_error5(discriminator_model):
    discriminator = discriminator_model

    with pytest.raises(Exception) as ex:
        SRGAN(input_shape_lr=(32, 32, 3),
              input_shape_hr=(128, 128, 3),
              num_residual_blocks=2,
              num_disc_blocks=2,
              discriminator=discriminator)

    assert "Both discriminator and generator should be passed, but only discriminator was found." in str(ex)
