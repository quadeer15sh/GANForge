import logging

import numpy as np
import pytest

from GANForge.losses import PerceptualLoss

logger = logging.getLogger()


def test_perceptual_loss_success():
    a = np.random.randn(2, 32, 32, 3)
    b = np.random.randn(2, 32, 32, 3)

    loss_fn = PerceptualLoss(activation_layer=2, weights=None)
    logger.info(f"Loss calculated successfully: {loss_fn(a, b)}")


def test_perceptual_loss_error1():
    a = np.random.randn(2, 32, 32, 1)
    b = np.random.randn(2, 32, 32, 1)

    with pytest.raises(AssertionError) as ex:
        loss_fn = PerceptualLoss(activation_layer=2, weights=None)
        loss_fn(a, b)

    assert f"perceptual loss can only take image tensor inputs with channels = 3, found channel 1" in str(ex)


def test_perceptual_loss_error2():

    with pytest.raises(ValueError) as ex:
        PerceptualLoss(activation_layer=21, weights=None)

    assert "VGG-19 cannot take activation layer below 1 or above 20, found 21" in str(ex)


def test_perceptual_loss_error3():

    with pytest.raises(ValueError) as ex:
        PerceptualLoss(activation_layer=0, weights=None)

    assert "VGG-19 cannot take activation layer below 1 or above 20, found 0" in str(ex)
