from GANForge.dcgan import DCGAN


def test_dcgan():
    model = DCGAN(input_shape=(28, 28, 3), latent_dim=100)
    assert model.latent_dim == 100
