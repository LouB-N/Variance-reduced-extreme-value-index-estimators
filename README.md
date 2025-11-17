 # Simulation Code for Variance-reduced ratio of means estimator using control variates in a semi-supervised setting
This repository contains the Python code used for the simulations in the paper:

> **Variance-reduced extreme value index estimators using control variates in a semi-supervised setting**  
> Louison Bocquet--Nouaille, Jérôme Morio, Benjamin Bobbia - 2025  



## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```


## Repository structure

The *src/experiments/compare_EVI_estimators.py* file provides code to run large-scale simulations in parallel, reducing computation time. It benchmarks classical EVI estimators (Hill, Moment, MLE) against variance-reduced versions (Transferred Hill, Transferred Moment, and SSE). The SSE method is adapted from **[Extreme Value Statistics in Semi-Supervised Models - Hanan Ahmed, John H.J. Einmahl, Chen Zhou - 2025]** (MIT License) which code is included in the file *src/functions/code_article_Ahmled_Einmahl_Zhou.R*, where the relevant R code was reorganized into a single `est()` function callable from Python.


The *src/functions* directory contains the different functions necessary for the experiments. In particular, the file *estimators.py* contains useful generic functions to compute the proposed variance-reduced estimators for any ratio of means.



## License

This code is released under the MIT license (see file LICENSE).