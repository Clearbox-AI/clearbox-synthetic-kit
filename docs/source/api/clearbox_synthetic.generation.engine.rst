Generative Engine
=================

The **Generative Engine** provides tools for creating synthetic data using advanced machine learning models. It includes modules tailored for time series and tabular data, leveraging **Variational Autoencoders (VAE)** and **Diffusion Models** to generate high-quality, privacy-preserving synthetic datasets.

The ``TimeSeriesEngine`` is designed for time series data generation, offering features like:

- Model training and evaluation.
- Latent space sampling.
- Reconstruction error analysis.

The ``TabularEngine`` handles tabular data, supporting both **VAE** and **Diffusion Models**. Key features include:

- Tools for training and evaluation.
- Sampling from latent space.
- Reconstruction of mixed data types.
- Configurable architectures for numerical and categorical data.

TabularEngine
~~~~~~~~~~~~~
.. automodule:: clearbox_synthetic.generation.engine.tabular_engine
   :members:
   :undoc-members:
   :show-inheritance:

TimeSeriesEngine
~~~~~~~~~~~~~~~~
.. automodule:: clearbox_synthetic.generation.engine.timeseries_engine
   :members:
   :undoc-members:
   :show-inheritance:
