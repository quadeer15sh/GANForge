import pytest

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, Dropout, Flatten, Dense, Input,
                                     BatchNormalization, Conv2DTranspose,
                                     LeakyReLU, Reshape, Embedding, Concatenate)
from GANForge.conditional_dcgan import ConditionalDCGAN


def test_conditional_dcgan_success1():
    model = ConditionalDCGAN(input_shape=(32, 32, 1), latent_dim=64, num_classes=2, class_embeddings_size=32)

    assert model.num_classes == 2
    assert model.latent_dim == 64
    assert model.class_embeddings_size == 32


def test_conditional_dcgan_success2():
    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(64 * 64)(x)
    label = Reshape((64, 64, 1))(x)

    input_image = Input(shape=(64, 64, 3))
    concat = Concatenate()([input_image, label])

    x = Conv2D(32, kernel_size=5, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)
    discriminator = Model(inputs=[input_image, input_label], outputs=[output])

    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(4 * 4)(x)
    label = Reshape((4, 4, 1))(x)

    input_latent = Input(shape=(256,))
    x = Dense(4 * 4 * 256)(input_latent)
    x = Reshape((4, 4, 256))(x)
    concat = Concatenate()([x, label])
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(x)
    generator = Model(inputs=[input_latent, input_label], outputs=[output])

    model = ConditionalDCGAN(num_classes=3, input_shape=(64, 64, 3), latent_dim=256, class_embeddings_size=32)
    assert model.generator.input_shape == generator.input_shape
    assert model.generator.output_shape == generator.output_shape
    assert model.discriminator.input_shape == discriminator.input_shape
    assert model.discriminator.output_shape == discriminator.output_shape


def test_conditional_dcgan_success3():
    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(64 * 64)(x)
    label = Reshape((64, 64, 1))(x)

    input_image = Input(shape=(64, 64, 3))
    concat = Concatenate()([input_image, label])

    x = Conv2D(32, kernel_size=5, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)
    discriminator = Model(inputs=[input_image, input_label], outputs=[output])

    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(4 * 4)(x)
    label = Reshape((4, 4, 1))(x)

    input_latent = Input(shape=(128,))
    x = Dense(4 * 4 * 256)(input_latent)
    x = Reshape((4, 4, 256))(x)
    concat = Concatenate()([x, label])
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(x)
    generator = Model(inputs=[input_latent, input_label], outputs=[output])

    model = ConditionalDCGAN(input_shape=(64, 64, 3), latent_dim=128,
                             num_classes=3, class_embeddings_size=32,
                             generator=generator, discriminator=discriminator)
    assert model.generator.input_shape == generator.input_shape
    assert model.generator.output_shape == generator.output_shape
    assert model.discriminator.input_shape == discriminator.input_shape
    assert model.discriminator.output_shape == discriminator.output_shape
    assert model.latent_dim == 128


def test_conditional_dcgan_success4():
    a = np.random.randn(2, 32, 32, 1)
    a[a < -1] = -1
    a[a > 1] = 1
    b = np.random.randint(0, 3, 2)

    model = ConditionalDCGAN(input_shape=(32, 32, 1), latent_dim=64, num_classes=3)
    model.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                  loss_fn=tf.keras.losses.BinaryCrossentropy())
    model.fit(a, b, epochs=1)


def test_conditional_dcgan_success5():
    latent_dim = 128
    random_latent_vectors = tf.random.normal(shape=(2, latent_dim))
    b = np.random.randint(0, 4, 2)
    model = ConditionalDCGAN(input_shape=(32, 32, 1), latent_dim=128, num_classes=4)
    output = model.generator.predict([random_latent_vectors, b])
    assert output.shape == (2, 32, 32, 1)


def test_conditional_dcgan_error1():
    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(64 * 64)(x)
    label = Reshape((64, 64, 1))(x)

    input_image = Input(shape=(64, 64, 3))
    concat = Concatenate()([input_image, label])

    x = Conv2D(32, kernel_size=5, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)
    discriminator = Model(inputs=[input_label, input_image], outputs=[output])

    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(4 * 4)(x)
    label = Reshape((4, 4, 1))(x)

    input_latent = Input(shape=(128,))
    x = Dense(4 * 4 * 256)(input_latent)
    x = Reshape((4, 4, 256))(x)
    concat = Concatenate()([x, label])
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(x)
    generator = Model(inputs=[input_latent, input_label], outputs=[output])
    with pytest.raises(Exception) as e_info:
        ConditionalDCGAN(input_shape=(64, 64, 3), latent_dim=128,
                         num_classes=3, class_embeddings_size=32,
                         generator=generator, discriminator=discriminator)

    assert "input_shape passed as input does not match the input_shape of the discriminator network" in str(e_info)


def test_conditional_dcgan_error2():
    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(64 * 64)(x)
    label = Reshape((64, 64, 1))(x)

    input_image = Input(shape=(64, 64, 3))
    concat = Concatenate()([input_image, label])

    x = Conv2D(32, kernel_size=5, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation="sigmoid")(x)
    discriminator = Model(inputs=[input_image, input_label], outputs=[output])

    input_label = Input(shape=(1,))
    x = Embedding(3, 32)(input_label)
    x = Dense(4 * 4)(x)
    label = Reshape((4, 4, 1))(x)

    input_latent = Input(shape=(128,))
    x = Dense(4 * 4 * 256)(input_latent)
    x = Reshape((4, 4, 256))(x)
    concat = Concatenate()([x, label])
    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(concat)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(3, kernel_size=5, padding="same", activation="tanh")(x)
    generator = Model(inputs=[input_label, input_latent], outputs=[output])
    with pytest.raises(Exception) as e_info:
        ConditionalDCGAN(input_shape=(64, 64, 3), latent_dim=128,
                         num_classes=3, class_embeddings_size=32,
                         generator=generator, discriminator=discriminator)

    assert "latent_dim passed as input does not match the latent_dim dimension of the generator model" in str(e_info)
