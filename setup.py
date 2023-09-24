import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="GANForge",
    version="0.0.1",
    author="Quadeer Shaikh",
    author_email="quadeershaikh15.8@gmail.com",
    license='MIT',
    description="Python library for a wide variety of GANs (Generative Adversarial Networks) based on TensorFlow and Keras.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url='https://github.com/quadeer15sh/GANForge',
    packages=setuptools.find_packages(),
    keywords=['GANs', 'tensorflow', 'keras'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    package_dir={'': 'src'},
    install_requires=['tensorflow', 'keras', 'matplotlib', 'seaborn']
)
