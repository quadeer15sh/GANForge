from dataclasses import dataclass
from typing import Union, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dropout, Flatten, Dense, Input,
                                     BatchNormalization, Conv2DTranspose,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Sequential, Model


@dataclass(frozen=True)
class _GANInputs:
    discriminator: Union[Sequential, Model]
    generator: Union[Sequential, Model]


class DCGAN(Model):
    """
    This module creates a Deep Convolutional General Adversarial Network (DCGAN) using Tensorflow. DCGAN is a class of CNNs that demonstrate how
    it can generate images in an unsupervised manner. This module follows the following guidelines and recommends the users to use the same settings
    in their hyperparameters selection

    1. Replace any pooling layers with strided convolutions in discriminator network and fractional strided convolutions in generator network
    2. Use batch normalization in both the generator and the discriminator
    3. Remove fully connected hidden layers for deeper architectures
    4. Use ReLU activation for all the layers in the generator except for the output layer which should use tanh
    5. Use Leaky ReLU activation in all the layers in the discriminator
    6. Use Adam optimizer with a learning rate of 0.0002 and momentum term beta1 of 0.5

    Reference: https://arxiv.org/pdf/1511.06434.pdf
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        discriminator: Optional[Union[Sequential, Model]] = None,
        generator: Optional[Union[Sequential, Model]] = None,
    ) -> None:
        """
        Creates a Tensorflow model for DCGAN. DCGAN model is created either using the default model configuration by providing
        the input_shape and latent_dim, or it can be created by passing a custom discriminator and generator.

        :param input_shape: input shape of the image in (height, width, channels) format. Example: (256, 256, 3). Recommended shapes for
                            default generator of DCGAN - 32x32, 64x64, 128x128, 160x160, 192x192, 224x224
        :param latent_dim: dimension of the latent vector using which images can be generated
        :param discriminator: discriminator network of the GAN. Note: the latent vector dim of the network should be the same as latent_dim
        :param generator: generator network of the GAN. Note: the input shape of the network should be the same as input_shape
        :raises: ValueError if only one of input_shape or latent_dim is passed or if only one of discriminator or generator is passed or if
                the input_shape does not match with the output shape of the default generator
        """
        super().__init__()
        _gan_inputs = self._get_gan_inputs(input_shape, latent_dim, discriminator, generator)
        self.latent_dim = latent_dim
        self._discriminator = _gan_inputs.discriminator
        self._generator = _gan_inputs.generator
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def generator(self):
        return self._generator

    def compile(
        self,
        d_optimizer,
        g_optimizer,
        loss_fn
    ) -> None:
        super().compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def _get_gan_inputs(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        discriminator: Union[Sequential, Model],
        generator: Union[Sequential, Model],
    ) -> _GANInputs:
        assert len(input_shape) == 3, "The input shape must be provided in (height, width, channels) format"

        if discriminator and generator:
            latent_dim_msg = ("Error: latent_dim passed as input does not match the latent_dim dimension of the generator model. Note: If your "
                              "model has multiple inputs then please ensure that the first input is for latent_dim")
            input_shape_msg = ("Error: input_shape passed as input does not match the input_shape of the discriminator network. Note: If you model "
                               "has multiple inputs then please ensure that the first input is for image input_shape")
            if type(generator) == Sequential:
                assert generator.layers[0].input_shape[1] == latent_dim, latent_dim_msg
                assert discriminator.layers[0].input_shape[1:] == input_shape, input_shape_msg
            else:
                assert generator.inputs[0].shape[1] == latent_dim, latent_dim_msg
                assert discriminator.inputs[0].shape[1:] == input_shape, input_shape_msg
            _discriminator = discriminator
            _generator = generator

        elif (generator is None) ^ (discriminator is None):
            passed = 'discriminator' if discriminator is not None else 'generator'
            raise ValueError(f"Both discriminator and generator should be passed, but only {passed} was found.")

        else:
            _discriminator = self._create_discriminator(input_shape)
            _generator = self._create_generator(input_shape, latent_dim, _discriminator)

            recommended_inputs = [32, 64, 96, 128, 160, 192, 224]
            if input_shape != _generator.output_shape[1:]:
                raise ValueError(f'''The input_shape does not match with the output shape of the default generator. Please provide 
                input_shape in one of the recommended shapes: {[(r, r, input_shape[-1]) for r in recommended_inputs]} or pass your own network inputs
                ''')

        return _GANInputs(discriminator=_discriminator, generator=_generator)

    def _create_discriminator(
        self,
        input_shape: Tuple[int, int, int]
    ) -> Sequential:
        return Sequential(
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

    def _create_generator(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        discriminator: Sequential
    ) -> Sequential:
        g_h = discriminator.layers[-7].output_shape[1]
        g_w = discriminator.layers[-7].output_shape[2]
        g_d = discriminator.layers[-7].output_shape[3]
        return Sequential(
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

    def summary(
        self
    ) -> None:
        self._discriminator.summary()
        self._generator.summary()

    def call(
        self,
        inputs
    ) -> tf.Tensor:
        pass

    @tf.function
    def train_step(
        self,
        real
    ) -> dict:
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake = self.generator(random_latent_vectors)

        with tf.GradientTape() as d_tape:
            loss_disc_real = self.loss_fn(tf.ones((batch_size, 1)), self.discriminator(real))
            loss_disc_fake = self.loss_fn(tf.zeros((batch_size, 1)), self.discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2

        grads = d_tape.gradient(loss_disc, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        with tf.GradientTape() as g_tape:
            fake = self.generator(random_latent_vectors)
            output = self.discriminator(fake)
            loss_gen = self.loss_fn(tf.ones(batch_size, 1), output)

        grads = g_tape.gradient(loss_gen, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_tracker.update_state(loss_disc)
        self.g_loss_tracker.update_state(loss_gen)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }
