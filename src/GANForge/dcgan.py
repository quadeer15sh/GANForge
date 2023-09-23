from typing import Union, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, Dropout, Flatten, Dense, Input,
                                     BatchNormalization, Conv2DTranspose,
                                     LeakyReLU, Reshape)


class DCGAN(Model):

    def __init__(
        self,
        input_shape: Optional[Tuple[int, int, int]] = None,
        latent_dim: Optional[int] = None,
        discriminator: Optional[Union[Sequential, Model]] = None,
        generator: Optional[Union[Sequential, Model]] = None,
    ) -> None:
        """
        Creates a Tensorflow model for DCGAN. DCGAN model is created either using the default model configuration by using
        the input_shape and latent_dim, or it can be created by passing a custom discriminator and generator.

        :param input_shape: input shape of the image in (height, width, channels) format. Example: (256, 256, 3)
        :param latent_dim: dimension of the latent vector using which images can be generated
        :param discriminator: discriminator network of the GAN
        :param generator: generator network of the GAN
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self._discriminator = Sequential(
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

        g_h, g_w, g_d = self._get_latent_dims()

        self._generator = Sequential(
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

        if (generator is None) ^ (discriminator is None):
            passed = 'discriminator' if discriminator is not None else 'generator'
            raise ValueError(f"Both discriminator and generator should be passed, but only {passed} was found.")

        assert len(input_shape) == 3, "The input shape must be provided in (height, width, channels) format"

        if discriminator and generator:
            self._discriminator = discriminator
            self._generator = generator

        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(
        self,
        d_optimizer,
        g_optimizer,
        loss_fn
    ):
        super().compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def _get_latent_dims(
        self
    ) -> tuple:
        g_h = self._discriminator.layers[-7].output_shape[1]
        g_w = self._discriminator.layers[-7].output_shape[2]
        g_d = self._discriminator.layers[-7].output_shape[3]

        return g_h, g_w, g_d

    def call(
        self,
        inputs
    ):
        result = self.generator(inputs)
        return result

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