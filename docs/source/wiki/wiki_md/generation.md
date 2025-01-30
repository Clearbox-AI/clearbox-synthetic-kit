<br>
<br>
<br>
The `Engine` and `Synthesizer` modules are the core of Clearbox Synthetic Kit. 🚀

These classes provide a comprehensive framework for training VAEs on **tabular** and **time-series data**, performing latent space encoding, and generating synthetic labeled/unlabeled datasets.\
By leveraging Variational Autoencoders (VAEs) and Diffusion Models, this system enables powerful data synthesis, privacy preservation, and unsupervised learning techniques in a flexible and scalable manner. 

This framework is particularly useful in applications such as:

- Data privacy: Creating synthetic data for analysis without exposing sensitive information.
- Anomaly detection: Identifying unusual data points using reconstruction errors.
- Data augmentation: Expanding datasets for training machine learning models.
- Feature learning: Extracting latent representations for clustering and classification.

#### 1. TabularEngine and TimeSeriesEngine

The `TabularEngine` class integrates a Variational Autoencoder and an optional Diffusion Model to process and generate tabular datasets. 

`TimeSeriesEngine` is dedicated to sequential data with dependencies across time steps and is specialized in training and evaluating transformer-based VAEs with attention mechanisms for time-series data.

Both Engines handle model initialization and training, evaluation, inference, and synthetic data sampling.\
They implement hyperparameter tuning and early stopping to optimize model performance and provide reconstruction and encoding functions for downstream machine learning tasks.\
They also allows latent space sampling to explore variations in the data before jumping to data synthesis.

The `train()` method starts the training of the generative model selected on tabular data, applying an optimizer and updating model parameters. The method supports early stopping and progress tracking.

#### 2. LabeledSynthesizer and UnlabeledSynthesizer

The Synthesizer classes are dedicated to generate brand-new synthetic data by leveraging a trained `TabularEngine` or `TimeSeriesEngine`.\
The `LabeledSynthesizer` is specialized in generating labeld data, while the `UnlabeledSynthesizer` generates synthetic data without labels.

They apply shuffling techniques and reconstruction-based sampling to create realistic synthetic samples while maintaining the statistical properties of the original dataset.

They include privacy-aware data synthesis to prevent data leakage. In fact, it is possible to add a controlled amount of noise to the latent space representation before sampling new data.\
This enables an iterative process between data generation and the evaluation of the privacy performance of synthetic data, ensuring that the privacy level meets the desired constraints.