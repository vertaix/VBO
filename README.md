# Vendi Bayesian Optimization and application to MOF design for NH₃ adsorption

This repository contains the implementation of the Vendi Bayesian Optimization (VBO), a Bayesian optimization algorithm that seeks to find diverse solutions to a black-box optimization problem, applied to a metal-organic framework (MOF) design space to optimize for MOF properties.
We do this by designing a MOF-specific kernel that accounts for a multitude of informative MOF properties, and using the [Vendi Score](https://github.com/vertaix/Vendi-Score) to encourage the candidate MOFs to be inspected to be diverse from one another.

<p align="center">
<img src="workflow.png" alt="workflow"/>
</p>

<p align="center">
<em>The workflow of our VBO framework, where diverse MOFs are iteratively selected to optimize NH₃ adsorption capabilities.</em>
</p>

For more information, please see our paper, [Diversity-driven, efficient exploration of a MOF design space to optimize MOF properties: application to NH₃ adsorption](https://chemrxiv.org/engage/chemrxiv/article-details/661dd99b418a5379b0ee73fc).

## Installation

You can clone this repository and install the necessary dependencies using the following commands:

```bash
conda create --name vbo
conda activate vbo
conda install python==3.11.3
pip install -r requirements.txt
```

## Usage

First download the data necessary to run our code at [this link](https://wustl.box.com/s/3jkz8ksu9l3d1hqikir4olainke9wc5t).
The downloaded `data` folder should be placed in the root directory of this repository, replacing the default `data` folder.

`mof_search/utils.py` contains the implementation of our Gaussian process model and other helper functions, which are used in `mof_search/run_bo.py` to facilitate an optimization run.

To start an optimization run, run `mof_search/run_bo.py` with the arguments of your choice.
- The `method` argument specifies the optimization method to run, supporting `VBO` (our method), `BO` (the traditional Bayesian optimization baseline), and `random` (random search).
- The `target` argument specifies the metric to be optimized, supporting `M_Storage`, `M_DBD`, and `M_safety`; for more details on these metrics, refer to Section 2.3 of [our paper](https://chemrxiv.org/engage/chemrxiv/article-details/661dd99b418a5379b0ee73fc).

An example command is shown below:
```bash
cd mof_search
python run_bo.py --method VBO --target M_Storage --seed 0
```

You can run the three methods with seeds 0–9 to replicate our experiment results.
You can further run the corresponding Jupyter notebooks in `notebooks` to recreate the figures in our paper.

## Citations
```bibtex
@article{liu2024diversity,
title={{Diversity-driven, efficient exploration of a MOF design space to optimize MOF properties: application to NH_3 adsorption}},
author={Liu, Tsung-Wei and Nguyen, Quan and Dieng, Adji Bousso and Gomez-Gualdron, Diego},
journal={ChemRxiv preprint},
year={2024}
}
```
