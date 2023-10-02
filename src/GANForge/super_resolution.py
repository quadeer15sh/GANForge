from dataclasses import dataclass
from typing import Tuple, Optional, Union

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Input,
                                     BatchNormalization, LeakyReLU, PReLU,
                                     GlobalAvgPool2D, Add)
from tensorflow.keras.models import Sequential, Model


@dataclass
class _GANInputs:
    discriminator: Union[Sequential, Model]
    generator: Union[Sequential, Model]


class SRGAN(Model):
    """
    Super-resolution (SR) is upsampling a low-resolution image into a higher resolution with minimal information distortion. The motive of SRGAN
    architecture is to recover finer textures from the image when it is up scaled without compromising its perceptual quality. SRGAN is trained
    using perceptual loss and adversarial loss as objective loss functions to convert a low resolution image to a high resolution image.

    Reference: https://arxiv.org/abs/1609.04802
    """
    def __init__(
        self,
        input_shape_lr: Tuple[int, int, int],
        input_shape_hr: Tuple[int, int, int],
        scaling_factor: int = 4,
        num_residual_blocks: int = 5,
        num_disc_blocks: int = 2,
        discriminator: Optional[Union[Sequential, Model]] = None,
        generator: Optional[Union[Sequential, Model]] = None,
    ) -> None:
        """
        Creates a Tensorflow model for Super Resolution GAN (SRGAN). The SRGAN model is created either using the default model
        configuration by providing the input shapes for low and high resolution images along with the scaling factor and other hyperparams,
        or it can be created by passing a custom discriminator and generator.

        :param input_shape_lr: input shape of the lower resolution image in (height, width, channels) format
        :param input_shape_hr: input shape of the higher resolution image in (height, width, channels) format
        :param scaling_factor: a scalar value which should be a multiple of low hi-res image dimensions with respect to the low-res image dimensions.
                               Note: This value currently defaults to and should be defaulted to 4, and will be worked upon in the upcoming updates
        :param num_residual_blocks: number of residual blocks in the generator
        :param num_disc_blocks: number of repeated blocks in the discriminator
        :param discriminator: SRGAN's discriminator network
        :param generator: SRGAN's generator network
        :raises: AssertionError if the input shape is not in (height, width, channels) format or if the hi-res image dimensions are not a multiple
                 of low-res image dimensions by a factor of scaling_factor
        """
        super().__init__()
        _gan_inputs = self._get_gan_inputs(input_shape_lr, input_shape_hr,
                                           scaling_factor, num_residual_blocks,
                                           num_disc_blocks, discriminator, generator)
        self._generator = _gan_inputs.generator
        self._discriminator = _gan_inputs.discriminator
        self.scaling_factor = scaling_factor
        self.num_residual_blocks = num_residual_blocks
        self.num_disc_blocks = num_disc_blocks
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_optimizer = None
        self.g_optimizer = None
        self.d_loss = None
        self.g_loss = None

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    def compile(
        self,
        d_optimizer,
        g_optimizer,
        d_loss,
        g_loss
    ) -> None:
        super().compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss = d_loss
        self.g_loss = g_loss

    @staticmethod
    def _validation_checks(input_shape_lr, input_shape_hr, scaling_factor):
        assert input_shape_hr[0] / input_shape_lr[0] == scaling_factor, (f"Error: The SRGAN model scales low resolution images to high resolution "
                                                                         f"by a factor of {scaling_factor}, "
                                                                         f"found: {int(input_shape_hr[0] / input_shape_lr[0])}")
        assert input_shape_hr[1] / input_shape_lr[1] == scaling_factor, (f"Error: The SRGAN model scales low resolution images to high resolution "
                                                                         f"by a factor of {scaling_factor}, "
                                                                         f"found: {input_shape_hr[1] / input_shape_lr[1]}")
        assert len(input_shape_lr) == 3, "The input shape must be provided in the following format (height, width, channels)"
        assert len(input_shape_hr) == 3, "The input shape must be provided in the following format (height, width, channels)"

    @staticmethod
    def _model_dimension_validation(model_, input_shape, error_message):
        if type(model_) == Sequential:
            assert model_.layers[0].input_shape[1:] == input_shape, error_message
        else:
            assert model_.inputs[0].shape[1:] == input_shape, error_message

    def _get_gan_inputs(
        self,
        input_shape_lr: Tuple[int, int, int],
        input_shape_hr: Tuple[int, int, int],
        scaling_factor: int,
        num_residual_blocks: int,
        num_disc_blocks: int,
        discriminator: Union[Sequential, Model],
        generator: Union[Sequential, Model]
    ) -> _GANInputs:
        SRGAN._validation_checks(input_shape_lr, input_shape_hr, scaling_factor)
        if discriminator and generator:
            lr_msg = "Error: input_shape_lr does not match the input shape of the generator"
            hr_msg = "Error: input_shape_hr does not match the input shape of the discriminator"

            self._model_dimension_validation(discriminator, input_shape_hr, hr_msg)
            self._model_dimension_validation(generator, input_shape_lr, lr_msg)

            _discriminator = discriminator
            _generator = generator

        elif (generator is None) ^ (discriminator is None):
            passed = 'discriminator' if discriminator is not None else 'generator'
            raise ValueError(f"Both discriminator and generator should be passed, but only {passed} was found.")

        else:
            _discriminator = self._create_discriminator(input_shape_hr, num_disc_blocks)
            _generator = self._create_generator(input_shape_lr, num_residual_blocks)

        return _GANInputs(discriminator=_discriminator, generator=_generator)

    @staticmethod
    def _create_discriminator(
        input_shape: Tuple[int, int, int],
        disc_blocks: int
    ) -> Sequential:
        discriminator = Sequential(name='discriminator')
        discriminator.add(Input(shape=input_shape))
        discriminator.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        discriminator.add(BatchNormalization())
        discriminator.add(LeakyReLU(0.2))

        for i in range(disc_blocks):
            discriminator.add(Conv2D(128 * (2 ** i), kernel_size=3, strides=1, padding="same"))
            discriminator.add(BatchNormalization())
            discriminator.add(LeakyReLU(0.2))
            discriminator.add(Conv2D(128 * (2 ** i), kernel_size=3, strides=2, padding="same"))
            discriminator.add(BatchNormalization())
            discriminator.add(LeakyReLU(0.2))

        discriminator.add(GlobalAvgPool2D())
        discriminator.add(Dense(1024))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dense(1, activation="sigmoid"))

        return discriminator

    @staticmethod
    def _create_generator(
        input_shape: Tuple[int, int, int],
        num_residual_blocks: int
    ) -> Model:
        input_image = Input(shape=input_shape)
        x = Conv2D(64, kernel_size=9, padding='same')(input_image)
        p_relu_output = PReLU(shared_axes=[1, 2])(x)
        x_in = p_relu_output

        for i in range(num_residual_blocks):
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

        return Model(inputs=input_image, outputs=output, name='generator')

    def call(
        self,
        inputs
    ):
        pass

    def summary(
        self
    ):
        self.discriminator.summary()
        self.generator.summary()

    @tf.function
    def train_step(
        self,
        images
    ):
        lr_images, hr_images = images
        batch_size = lr_images.shape[0]
        sr_images = self.generator(lr_images)

        with tf.GradientTape() as d_tape:
            loss_disc_hr = self.d_loss(tf.ones((batch_size, 1)), self.discriminator(hr_images))
            loss_disc_sr = self.d_loss(tf.zeros((batch_size, 1)), self.discriminator(sr_images))
            loss_disc = (loss_disc_hr + loss_disc_sr)/2

        grads = d_tape.gradient(loss_disc, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        with tf.GradientTape() as g_tape:
            sr_images = self.generator(lr_images)
            output = self.discriminator(sr_images)
            adversarial_loss = 1e-3 * self.d_loss(tf.ones(batch_size, 1), output)
            perceptual_loss = self.g_loss(hr_images, sr_images)

            loss_gen = perceptual_loss + adversarial_loss

        grads = g_tape.gradient(loss_gen, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_tracker.update_state(loss_disc)
        self.g_loss_tracker.update_state(loss_gen)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }
