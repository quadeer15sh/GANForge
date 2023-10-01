from typing import Union, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dropout, Flatten, Dense, Input,
                                     BatchNormalization, Conv2DTranspose,
                                     LeakyReLU, Reshape, Embedding, Concatenate)
from tensorflow.keras.models import Sequential, Model

from GANForge.dcgan import DCGAN


class ConditionalDCGAN(DCGAN):
    """
    This module creates a Conditional Deep Convolutional General Adversarial Network (cDCGAN) using Tensorflow. cDCGAN is a type of GAN that
    involves the conditional generation of images. In cDCGANs the conditional setting is applied in a way such that both the generator and
    discriminator are conditioned on some sort of auxiliary information such as class labels. As a result this GAN can learn multiple modes of
    mapping from inputs to outputs by being fed with different contextual information in the form of class labels.

    Reference: https://arxiv.org/pdf/1411.1784.pdf
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        num_classes: int,
        class_embeddings_size: Optional[int] = 50,
        discriminator: Optional[Union[Sequential, Model]] = None,
        generator: Optional[Union[Sequential, Model]] = None,
    ) -> None:
        """
        Creates a Tensorflow model for Conditional DCGAN. The Conditional DCGAN model is created either using the default model configuration by
        providing the num_classes, input_shape and latent_dim, or it can be created by passing a custom discriminator and generator.

        :param num_classes: Number of classes in the dataset used in the Conditional DCGAN model
        :param input_shape: input shape of the image in (height, width, channels) format. Example: (256, 256, 3). Recommended shapes for
                            default generator of DCGAN - 32x32, 64x64, 128x128, 160x160, 192x192, 224x224
        :param latent_dim: dimension of the latent vector using which images can be generated
        :param class_embeddings_size: Embedding dimension for creating a vector projection of classes
        :param discriminator: discriminator network of the GAN. Note: the latent vector dim of the network should be the same as latent_dim
        :param generator: generator network of the GAN. Note: the input shape of the network should be the same as input_shape
        :raises: ValueError if only one of input_shape or latent_dim is passed or if only one of discriminator or generator is passed or if
                 the input_shape does not match with the output shape of the default generator
        """
        self.num_classes = num_classes
        self.class_embeddings_size = class_embeddings_size
        super().__init__(input_shape, latent_dim, discriminator, generator)

    def _create_discriminator(
        self,
        input_shape: Tuple[int, int, int]
    ) -> Model:
        input_label = Input(shape=(1,))
        x = Embedding(self.num_classes, 50)(input_label)
        x = Dense(input_shape[0] * input_shape[1])(x)
        label = Reshape((input_shape[0], input_shape[1], 1))(x)

        input_image = Input(shape=input_shape)
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

        return Model(inputs=[input_image, input_label], outputs=[output])

    def _create_generator(
        self,
        input_shape: Tuple[int, int, int],
        latent_dim: int,
        discriminator: Sequential
    ) -> Model:
        g_h = discriminator.layers[-7].output_shape[1]
        g_w = discriminator.layers[-7].output_shape[2]
        g_d = discriminator.layers[-7].output_shape[3]

        input_label = Input(shape=(1,))
        x = Embedding(self.num_classes, 50)(input_label)
        x = Dense(g_h * g_w)(x)
        label = Reshape((g_h, g_w, 1))(x)

        input_latent = Input(shape=(latent_dim,))
        x = Dense(g_h * g_w * g_d)(input_latent)
        x = Reshape((g_h, g_w, g_d))(x)
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
        output = Conv2D(input_shape[2], kernel_size=5, padding="same", activation="tanh")(x)

        return Model(inputs=[input_latent, input_label], outputs=[output])

    @tf.function
    def train_step(
        self,
        data
    ) -> dict:
        real, label = data

        if len(label.shape) > 1:
            label = tf.math.argmax(label, axis=1)

        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake = self.generator([random_latent_vectors, label])

        with tf.GradientTape() as d_tape:
            loss_disc_real = self.loss_fn(tf.ones((batch_size, 1)), self.discriminator([real, label]))
            loss_disc_fake = self.loss_fn(tf.zeros((batch_size, 1)), self.discriminator([fake, label]))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

        grads = d_tape.gradient(loss_disc, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        with tf.GradientTape() as g_tape:
            fake = self.generator([random_latent_vectors, label])
            output = self.discriminator([fake, label])
            loss_gen = self.loss_fn(tf.ones(batch_size, 1), output)

        grads = g_tape.gradient(loss_gen, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_tracker.update_state(loss_disc)
        self.g_loss_tracker.update_state(loss_gen)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }
