# GANForge

Python library for a wide variety of GANs (Generative Adversarial Networks) based on TensorFlow and Keras.

# Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Examples](#examples)
4. [Supported GANs](#supported-gans)
5. [Custom Callbacks](#custom-callbacks)

# Installation <a id="installation"></a>

To download the GANForge model from pypi please use the following pip command in your 
command prompt/terminal
```
pip install GANForge
```

# Quick Start 

You can get started with building GANs in just a few lines of code, here is an example.

```python
import tensorflow as tf
from GANForge.dcgan import DCGAN

# train_ds: your image dataset

model = DCGAN(input_shape=(64, 64, 3), latent_dim=128)
model.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss_fn=tf.keras.losses.BinaryCrossentropy())

model.fit(train_ds, epochs=25)
```

# Examples

Please feel free to explore through the notebook files on each of the GAN models available in GANForge
- [Jupyter Notebooks](examples/README.md)
- [Kaggle Implementations]()

# Supported GANs

**Note :** This list is updated frequently, please come back to check if the GAN architecture you desire to use
is available or not

| Sr. | GAN Architecture | Status                                        |  
|-----|------------------|-----------------------------------------------|  
| 1   | DC GAN           | <span style="color:green">Available</span>    |  
| 3   | Conditional GAN  | <span style="color:orange">In Progress</span> |
| 4   | Info GAN         | <span style="color:orange">In Progress</span> |
| 5   | Pix2Pix GAN      | <span style="color:orange">In Progress</span> |
| 6   | Cycle GAN        | <span style="color:orange">In Progress</span> |
| 7   | Attention GAN    | <span style="color:orange">In Progress</span> |

# Custom Callbacks

Custom callbacks available for usage during your training

| Sr. | Callback           | GAN Applicable |  
|-----|--------------------|----------------|  
| 1   | DCGANVisualization | DC GAN         |  

### Example: 
```python
import tensorflow as tf
from GANForge.dcgan import DCGAN
from GANForge.callbacks import DCGANVisualization

# train_ds: your image dataset

model = DCGAN(input_shape=(64, 64, 3), latent_dim=128)
model.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
              loss_fn=tf.keras.losses.BinaryCrossentropy())
visualizer = DCGANVisualization(n_epochs=5)

model.fit(train_ds, epochs=25, callbacks=[visualizer])
```
