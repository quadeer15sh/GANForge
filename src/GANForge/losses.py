import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


def _get_vgg19():
    model = VGG19(include_top=False, input_shape=(None, None, 3), weights='imagenet')
    return Model(inputs=model.inputs, outputs=model.layers[20].output)


VGG_19 = _get_vgg19()


def perceptual_loss(
    hr_image: tf.Tensor,
    sr_image: tf.Tensor
):
    hr_preprocessed = preprocess_input(hr_image)
    sr_preprocessed = preprocess_input(sr_image)

    hr_featuremap = VGG_19(hr_preprocessed) / 12.75
    sr_featuremap = VGG_19(sr_preprocessed) / 12.75

    return tf.reduce_mean(tf.square(hr_featuremap - sr_featuremap))
