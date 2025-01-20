Synthesizer
===========

This section introduces tools for generating synthetic data, both labeled and unlabeled, from a pre-trained Engine instance while adhering to specific constraints and ensuring privacy compliance. The synthesizers provide methods for shuffling, reconstruction, and hybrid column handling, making them ideal for augmenting datasets in various machine learning applications.

The ``LabeledSynthesizer`` generates synthetic labeled data by:

- Sampling and reshuffling original data points.
- Supporting hybrid columns for combined feature processing.
- Incorporating latent space noise for variability.
- Adhering to column-specific rules and constraints.

The ``UnlabeledSynthesizer`` focuses on generating synthetic unlabeled data by:

- Leveraging latent space encoding and decoding.
- Sampling based on similarity in encoded representation.
- Supporting hybrid feature columns and custom rules.

UnlabeledSynthesizer
~~~~~~~~~~~~~~~~~~~~
.. automodule:: clearbox_synthetic.generation.synthesizer.unlabeled_synthesizer
   :members:
   :undoc-members:
   :show-inheritance:

LabeledSynthesizer
~~~~~~~~~~~~~~~~~~
.. automodule:: clearbox_synthetic.generation.synthesizer.labeled_synthesizer
   :members:
   :undoc-members:
   :show-inheritance:


