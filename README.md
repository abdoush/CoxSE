<h1 align="center">CoxSE & CoxSENAM</h1>
<p align="center"><em>Self-Explaining Neural Networks for Survival Analysis</em></p>
This repository contains the official implementation of CoxSE and CoxSENAM, two interpretable neural network architectures for survival analysis


This repository accompanies the article:

A. Alabdallah, O. Hamed, M. Ohlsson, T. Rögnvaldsson, S. Pashami. **CoxSE: Exploring the Potential of Self-Explaining Neural Networks with Cox Proportional Hazards Model for Survival Analysis**. *Knowledge-Based Systems* 2025, 114996, https://doi.org/10.1016/j.knosys.2025.114996  



## Using the Models
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Run **Example.ipybn** notebook to train CoxSE and CoxSENAM on a sample dataset and visualize explanations

3. This repository additionally provides implementations of the following models:
   1. [CoxNAM](https://www.sciencedirect.com/science/article/abs/pii/S0957417423007200): Cox-based Neural Additive Model.
   2. [DeepSurv](https://link.springer.com/article/10.1186/s12874-018-0482-1): Cox-based black-box deep survival analysis model.
   3. [CPH](https://www.jstor.org/stable/2985181): Cox Proportional Hazards model trained using gradient descent.

## Citation
If you use CoxSE or CoxSENAM in your research, please cite the following paper:
## BibTeX 
```
@article{ALABDALLAH2025114996,
    title = {CoxSE: Exploring the Potential of Self-Explaining Neural Networks with Cox Proportional Hazards Model for Survival Analysis},
    journal = {Knowledge-Based Systems},
    pages = {114996},
    year = {2025},
    issn = {0950-7051},
    doi = {https://doi.org/10.1016/j.knosys.2025.114996},
    url = {https://www.sciencedirect.com/science/article/pii/S0950705125020349},
    author = {Abdallah Alabdallah and Omar Hamed and Mattias Ohlsson and Thorsteinn Rögnvaldsson and Sepideh Pashami},
    keywords = {Self-Explaining Neural Networks, Cox Proportional Hazards, Survival Analysis, Interpretability, XAI, Neural Additive Models}
}
}
```


