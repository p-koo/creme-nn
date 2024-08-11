Welcome to CREME: Cis-Regulatory Element Model Explanations
===========================================================

.. image:: ../../img/creme_overview.png
   :alt: CREME Overview
   :width: 100%
   :align: center

CREME is a powerful toolkit for interpreting deep neural networks (DNNs) trained on genomic data. It helps researchers understand how these models predict gene expression and identify important regulatory elements in DNA sequences.

Key Features
------------

- Identify cis-regulatory elements (CREs) that directly enhance or silence target genes
- Map CRE distance from transcription start sites and gene expression
- Analyze the intricate complexity of higher-order CRE interactions
- Treat trained DNNs as surrogates for experimental assays, enabling in silico "measurements" for any sequence

Quick Start
-----------

Install CREME using pip:

.. code-block:: bash

   pip install creme-nn

Then, import and use CREME in your Python scripts:

.. code-block:: python

   from creme import context_dependence_test

   # Example usage
   results = context_dependence_test(model, sequence, tile_pos=[100, 200], num_shuffle=10)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: CREME Functions

   usage

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/adding_a_custom_model
   tutorials/context_dependence
   tutorials/context_swap
   tutorials/necessity_test
   tutorials/sufficiency_test_and_fine_tile_search
   tutorials/distance_test
   tutorials/higher_order_interaction_test
   tutorials/multiplicity_test

About CREME
-----------

CREME works by performing various in silico perturbation experiments on DNA sequences and analyzing how these changes affect the DNN's predictions. This approach mimics wet-lab experiments but allows for faster, more comprehensive testing of hypotheses about gene regulation.

Who is it for?
~~~~~~~~~~~~~~

CREME is designed for computational biologists, bioinformaticians, and machine learning researchers working on genomics problems. It's particularly useful for those studying gene regulation, enhancer-promoter interactions, and the effects of genetic variations.

Citing CREME
~~~~~~~~~~~~

If you use CREME in your research, please cite our paper:

.. code-block:: text

   Toneyan S, Koo PK. Interpreting cis-regulatory interactions from large-scale deep neural networks for genomics. bioRxiv. 2023.

Get Involved
------------

- `GitHub Repository <https://github.com/p-koo/creme-nn>`_
- `Report Issues <https://github.com/p-koo/creme-nn/issues>`_
- `Contribute <https://github.com/p-koo/creme-nn/blob/master/CONTRIBUTING.md>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
