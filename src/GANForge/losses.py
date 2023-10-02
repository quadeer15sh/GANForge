from typing import Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


class PerceptualLoss(Loss):

    def __init__(
        self,
        activation_layer: Optional[int] = 20,
        weights: Optional[str] = 'imagenet'
    ) -> None:
        """
        It measures the difference between the high-level features of two images, typically extracted from a pre-trained CNN like VGG-19

        :param activation_layer: layer from which feature maps need to be extracted
        :param weights: weights of the pre-trained CNN: VGG-19
        """
        super().__init__()
        model = VGG19(include_top=False, input_shape=(None, None, 3), weights=weights)
        self.vgg = Model(inputs=model.inputs, outputs=model.layers[activation_layer].output)

    def call(
        self,
        hr_image: tf.Tensor,
        sr_image: tf.Tensor
    ):

        assert hr_image.shape[-1] == 3, f"perceptual loss can only take image tensor inputs with channels = 3, found channel {hr_image.shape[-1]}"
        assert sr_image.shape[-1] == 3, f"perceptual loss can only take image tensor inputs with channels = 3, found channel {sr_image.shape[-1]}"

        hr_preprocessed = preprocess_input(hr_image)
        sr_preprocessed = preprocess_input(sr_image)

        hr_feature_map = self.vgg(hr_preprocessed) / 12.75
        sr_feature_map = self.vgg(sr_preprocessed) / 12.75

        return tf.reduce_mean(tf.square(hr_feature_map - sr_feature_map))
