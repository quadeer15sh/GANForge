from typing import Union, Optional, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, Dropout, Flatten, Dense, Input,
                                     BatchNormalization, Conv2DTranspose,
                                     LeakyReLU, Reshape)


@dataclass(frozen=True)
class _GANInputs:
    latent_dim: Optional[int] = None
    discriminator: Optional[Union[Sequential, Model]] = None
    generator: Optional[Union[Sequential, Model]] = None


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

        :param input_shape: input shape of the image in (height, width, channels) format. Example: (256, 256, 3). Recommended shapes for
                            default generator of DCGAN - 32x32, 64x64, 128x128, 160x160, 192x192, 224x224
        :param latent_dim: dimension of the latent vector using which images can be generated
        :param discriminator: discriminator network of the GAN
        :param generator: generator network of the GAN
        :raise: ValueError if only one of input_shape or latent_dim is passed or if only one of discriminator or generator is passed or if
                the input_shape does not match with the output shape of the default generator
        """
        super().__init__()
        _gan_inputs = self.__get_gan_inputs(input_shape, latent_dim, discriminator, generator)
        self.latent_dim = _gan_inputs.latent_dim
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

    @staticmethod
    def __get_gan_inputs(
        input_shape: Optional[Tuple[int, int, int]] = None,
        latent_dim: Optional[int] = None,
        discriminator: Optional[Union[Sequential, Model]] = None,
        generator: Optional[Union[Sequential, Model]] = None,
    ) -> _GANInputs:
        if discriminator and generator:
            _discriminator = discriminator
            _generator = generator
            _latent_dim = generator.layers[0].input_shape[1]
        elif (generator is None) ^ (discriminator is None):
            passed = 'discriminator' if discriminator is not None else 'generator'
            raise ValueError(f"Both discriminator and generator should be passed, but only {passed} was found.")
        else:
            if (input_shape is None) ^ (latent_dim is None):
                passed = 'input_shape' if input_shape is not None else 'latent_dim'
                raise ValueError(f"Both input_shape and latent_dim should be passed, but only {passed} was found.")
            assert len(input_shape) == 3, "The input shape must be provided in (height, width, channels) format"
            _latent_dim = latent_dim
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

            recommended_inputs = [32, 64, 96, 128, 160, 192, 224]
            if input_shape != _generator.output_shape[1:]:
                raise ValueError(f'''The input_shape does not match with the output shape of the default generator. Please provide 
                input_shape in one of the recommended shapes: {[(r, r, input_shape[-1]) for r in recommended_inputs]} or pass your own network inputs
                ''')

        return _GANInputs(latent_dim=_latent_dim, discriminator=_discriminator, generator=_generator)

    def summary(
        self
    ) -> str:
        return f"{self._discriminator.summary()}\n{self._generator.summary()}"

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
