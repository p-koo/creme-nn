# CREME: Cis-Regulatory Element Model Explanations

<img src="img/creme_overview.png" alt="CREME Overview" width="900"/>

[CREME](https://www.youtube.com/watch?v=PBwAxmrE194) is an advanced in silico perturbation framework designed to examine large-scale Deep Neural Networks (DNNs) trained on regulatory genomics data. CREME provides interpretations at various scales, from a coarse-grained CRE-level view to a fine-grained motif-level view. It is compatible with any ML framework, including TensorFlow and PyTorch. 

## Key Features

- Identify cis-regulatory elements (CREs) that directly enhance or silence target genes
- Map CRE distance from transcription start sites and gene expression
- Analyze the intricate complexity of higher-order CRE interactions
- Treat trained DNNs as surrogates for experimental assays, enabling in silico "measurements" for any sequence

## Core Functions

1. `context_dependence_test`: Examines how sequence patterns behave in different background contexts. Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/context_dependence.html).
2. `context_swap_test`: Analyzes the effect of placing a source sequence pattern in a target sequence context. Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/context_swap.html).
3. `necessity_test`: Measures the impact of tile shuffles on model predictions. Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/necessity_test.html).
4. `sufficiency_test`: Determines if a region of the sequence, along with the TSS tile, is sufficient for model predictions. Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/sufficiency_test_and_fine_tile_search.html).
5. `distance_test`: Maps the distance dependence between two tiles (one anchored, one variable). Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/distance_test.html).
6. `higher_order_interaction_test`: Performs a greedy search to identify optimal tile sets for changing model predictions. Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/higher_order_interaction_test.html).
7. `multiplicity_test`: Examines the effect of multiple copies of a CRE on model predictions.
8. `prune_sequence`: Also called Fine-tile search. Optimizes a tile through greedy search to find a sufficient subset of the most enhancing sub-tiles. Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/sufficiency_test_and_fine_tile_search.html#Fine-tile-search).
9. `Adding a custom model to use with CREME`: Tutorial [here](https://creme-nn.readthedocs.io/en/latest/tutorials/adding_a_custom_model.html).

## Installation

CREME is pip installable:

```bash
pip install creme-nn
```

## Dependencies

```
pyranges==0.0.120
pandas==2.0.1
seaborn==0.13.2
numpy==1.23.5
matplotlib==3.7.5
tqdm==4.65.0
tensorflow-gpu==2.11.1  # to replicate Enformer and Borzoi
tensorflow-hub==0.13.0  # to replicate Enformer and Borzoi
natsort==8.3.1
pyfaidx==0.7.2.1
kipoiseq==0.7.1
logomaker==0.8
```

## Usage
For detailed usage instructions and examples, please refer to our documentation and tutorials.
Colab Examples

Colab examples:
* Examples of CREME tests with Enformer:
    * https://colab.research.google.com/drive/1j3vXKf4QNgCWoIp655ugxEGyBN0cp4K5?usp=sharing
* Example of CREME with a PyTorch version of Enformer:
    * https://colab.research.google.com/drive/1c0ac3ei4Ntx0AgTaRkr80O8wZNb-j6wu?usp=sharing

## Resources

#### Paper: https://www.biorxiv.org/content/10.1101/2023.07.03.547592v1

#### Full documentation on [Readthedocs.org](https://creme-nn.readthedocs.io/en/latest/index.html)!

#### Tutorials: https://creme-nn.readthedocs.io/en/latest/tutorials.html

#### Results to replicate paper with intermediate results: https://zenodo.org/records/12584210 


## Citation
If you use CREME in your research, please cite our paper:
Toneyan S, Koo PK. Interpreting cis-regulatory interactions from large-scale deep neural networks for genomics. bioRxiv. 2023 Jul 3.

## License
[MIT License](https://github.com/p-koo/creme-nn/blob/master/LICENSE)

## Contact
email: koo@cshl.edu 
