from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class DCGANVisualization(Callback):
    def __init__(
        self,
        n_epochs: Optional[int] = 5
    ) -> None:
        """
        Displays 10 randomly generated images using DCGAN per n_epochs

        :param n_epochs: number of epochs after which output visualization is displayed
        :raises: ValueError if used for any other GANForge model other than DCGAN
        """
        super().__init__()
        self.n_epochs = n_epochs

    def on_epoch_end(
        self,
        epoch: int,
        logs=None
    ) -> None:
        if type(self.model).__name__ != 'DCGAN':
            raise ValueError(f"Invoked for model {type(self.model).__name__} ! This callback is available only for DCGAN models")

        if (epoch+1) % self.n_epochs == 0:
            print(f"\nEpoch: {epoch+1}, Generated images from randomly sampled latent vectors\n")
            latent_dim = self.model.latent_dim
            random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
            fake = self.model.generator(random_latent_vectors)
            generated_images = fake.numpy()
            plt.figure(figsize=(20, 6))
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                plt.subplots_adjust(hspace=0.5, wspace=0.3)
                image = generated_images[i]
                plt.imshow((image + 1) / 2)
                plt.axis('off')
            plt.show()


class ConditionalGANVisualization(Callback):
    def __init__(
        self,
        labels: List,
        n_epochs: Optional[int] = 5,
    ) -> None:
        """
        Displays 10 randomly generated images using DCGAN per n_epochs

        :param n_epochs: number of epochs after which output visualization is displayed
        :raises: ValueError if used for any other GANForge model other than DCGAN
        """
        super().__init__()
        self.labels = labels
        self.n_epochs = n_epochs

    def on_epoch_end(
        self,
        epoch: int,
        logs=None
    ) -> None:
        if type(self.model).__name__ != 'ConditionalDCGAN':
            raise ValueError(f"Invoked for model {type(self.model).__name__} ! This callback is available only for Conditional DCGAN models")

        if (epoch+1) % self.n_epochs == 0:
            print(f"\nEpoch: {epoch+1}, Generated images from randomly sampled latent vectors and labels\n")
            latent_dim = self.model.latent_dim
            class_labels = np.random.randint(0, self.model.num_classes, 10)
            random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
            fake = self.model.generator([random_latent_vectors, class_labels])
            generated_images = fake.numpy()
            plt.figure(figsize=(20, 6))
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                plt.subplots_adjust(hspace=0.5, wspace=0.3)
                image = generated_images[i]
                plt.imshow((image + 1) / 2)
                plt.title(f"Class: {self.labels[class_labels[i]]}")
                plt.axis('off')
            plt.show()
