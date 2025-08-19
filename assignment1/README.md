<br />

<img src = "../images/AIAP-Banner.png">

---

# `Assignment 1:` Unsupervised Learning

## Overview

This assignment introduces __unsupervised learning__, specifically techniques for clustering and dimensionality reduction. As you work through the two parts of the assignment, you will find yourself generating many intermediate datasets, trying different models, and tuning each model as you go along. There's a lot to keep track of.

Thus, this assignment also introduces ideas of __workflow management__ and __computational reproducibility__. Workflow management means organising your project directory to manage your analysis' artefacts (visualisations, processed datasets, notebooks, utility functions and experiment results). Ideally, your code should be clearly commented and variables should have well chosen names. 

Computational reproducibility means someone else (including future you!) should be able to take just the code and data, and reproduce your project, from its results and models to visualisations. 

How one decides to practice workflow management and computational reproducibility can be quite a personal decision. Therefore, we provide guidelines, not rules. The most important is having a system.

---

## Deliverables

For this assignment, there are three deliverables.

1. Edited Jupyter Notebooks with your workings.

2. A function in a Python module that automates the data cleaning and feature engineering steps you take so your findings are reproducible.

3. An exported record of software package versions and software environment used i.e. a `conda.yml` file. *(Psst! This [cheatsheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) might bring some enlightenment.)* Note that you __should not__ use the command to automatically generate the `conda.yml` file. Instead, you should manually generate the file by listing the packages __and__ their versions. The automatic generation will include some system specific information which will cause the automatic tests to fail.

---

### Part 1: Data Cleaning & Feature Engineering

__File:__ ['A1P1_data_cleaning_feature_engineering.ipynb'](A1P1_data_cleaning_feature_engineering.ipynb)

In the first part of this assignment, you explore the dataset that you'll be working with for this assignment.

---

### Part 2: Clustering

__File:__ ['A1P2_clustering.ipynb'](A1P2_clustering.ipynb)

After carrying out the tasks in Part 1, you will begin exploring several clustering methods and how do they differ from each other.

---
