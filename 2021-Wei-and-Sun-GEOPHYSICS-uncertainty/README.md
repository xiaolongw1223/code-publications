**[abstract](#Abstract) | [contents](#Contents) | [usage](#Usage) | [running the notebooks](#running-the-notebooks) | [issues](#issues) | [citation](#citation) | [license](#license)**

# Uncertainty analysis of 3D potential-field deterministic inversion using mixed Lp norms

This repo can be used to reproduce inversion results from the paper published on GEOPHYSICS: [(Wei and Sun, 2021)](https://doi.org/10.1190/geo2020-0672.1). 

## Abstract 

The non-uniqueness problem in geophysical inversion, especially potential-field inversion, is widely recognized. It is argued that uncertainty analysis of a recovered model should be as important as finding an optimal model. However, quantifying uncertainty still remains challenging, especially for $3$D inversions in both deterministic and Bayesian frameworks. Our objective is to develop an efficient method to empirically quantify the uncertainty of the physical property models recovered from $3$D potential-field inversion. We worked in a deterministic framework where an objective function consisting of a data misfit term and a regularization term is minimized. We performed inversions using a mixed $L_p$-norm formulation where various combinations of $L_p$ ($0\leq p\leq2$) norms can be implemented on different components of the regularization term. Specifically, we randomly sampled the $p$-norm values in multiple times, and generated a large and diverse sequence of physical property models that all reproduce the observed geophysical data equally well. This suite of models offers practical insights into the uncertainty of the recovered model features. We quantified the uncertainty through calculation of standard deviations and interquartile range, as well as visualizations in box plots and histograms. The numerical results for a realistic synthetic density model created based on a ring-shaped igneous intrusive body quantitatively illustrate uncertainty reduction due to different amounts of prior information imposed on inversions. We also applied the method to a field data set over the Decorah area in the northeastern Iowa. We adopted an acceptance-rejection strategy to generate 31 equivalent models based on which the uncertainties of the inverted models as well as the volume and mass estimates are quantified.

## Contents

There are two python scripts in this repository:

1. [grav_inv_irls](./grav_inv_irls.py): This script can be used to perform mixed Lp norm inversion.
2. [plt](./plt.py): This script can be use to visualize data maps as well as inverted model.

There are also several accompanying files that will be loaded into the code:

1. [mesh](./mesh.txt): UBC mesh file
2. [grav](./grav.obs): observed gravity gradient data simulated by using a horseshoe shaped synthetic model.
3. [topo](./topo.topo): topography

## Usage

Here are step-by-step instructions for running these notebooks locally on your machine:

Install Python. You can use [anaconda](https://www.anaconda.com/download/) for this.

- Install dependencies:
```
pip install -r requirements.txt
```

- Or, set up working environment using conda:
```
conda env create -f environment.yml
conda activate sparse-environment
```

- Install SimPEG
```
pip install SimPEG
```
- Or
```
conda install SimPEG --channel conda-forge
```

## Running the code

You can run the code in Spyder by clicking  `Run file` in the toolbar.

## Issues

Please [make an issue](https://github.com/simpeg-research/Wei-and-Sun-Uncertainty-2021/issues) if you encounter any problems while trying to run the notebooks.

## Citation

To cite this work, please reference

Wei, X. and Sun, J., 2021. Uncertainty analysis of 3D potential-field deterministic inversion using mixed L p norms. Geophysics, 86(6), pp.1-103.

## License
These notebooks are licensed under the [MIT License](/LICENSE) which allows academic and commercial re-use and adaptation of this work.

