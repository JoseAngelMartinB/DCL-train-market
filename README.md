<a name="readme-top"></a>
![Python version](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-blue)
![Conda](https://img.shields.io/badge/environment-conda-blue?logo=anaconda)
![Reproducibility](https://img.shields.io/badge/reproducibility-yes-brightgreen)
<!-- 
![Project Status](https://img.shields.io/badge/status-published-brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-YYYY.NNNNN-b31b1b.svg)](https://arxiv.org/abs/YYYY.NNNNN)
-->

# Stochastic Optimization through Distributional Constraint Learning (DCL) for a Pricing Problem in Competitive Rail Markets

Authors:
* José Ángel Martín-Baos
* Antonio Alcántara
* Carlos Ruiz
* Ricardo García Ródenas

These codes are associated with the paper "Optimization with constraint learning for pricing services under competition". This paper can be downloaded from ArXiv at [-----](https://arxiv.org/abs/YYYY.NNNNN).

If you use any part of the code or data provided in this repository, please cite it as:
> José Ángel Martín-Baos, Antonio Alcántara, Carlos Ruiz, Ricardo García-Ródenas (2025). Optimization with constraint learning for pricing services under competition. ArXiv preprint. https://arxiv.org/abs/YYYY.NNNNN


## Software implementation

All source code used to generate the results and figures in the paper are contained in this repository. The code is written in Python 3.11, and is organised in the following folders:

The folder `DataGenerationROBIN` contains the code used to generate the synthetic dataset using the [ROBIN (Rail mOBIlity simulatioN)](https://github.com/JoseAngelMartinB/robin) python package.

The `ConstraintLearning` folder contains the code necessary for training the machine learning models and executing the optimization with constraint learning models defined in this paper.

The root folder contains the environment file with the Anaconda/Mamba dependencies needed to execute the code.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/JoseAngelMartinB/DCL-train-market.git

or [download a zip archive](https://github.com/JoseAngelMartinB/DCL-train-market/archive/master.zip).


## Dependencies

You will need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create -f environment.yml


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate dcl-trainmarket

or, if you're on Windows:

    activate dcl-trainmarket

This will enable the environment for your current terminal session.
Any subsequent commands will use software that is installed in the environment.

To execute the Jupyter notebooks you must first start the notebook server by going into the
repository top level and running:

    jupyter notebook

This will start the server and open your default web browser to the Jupyter
interface. In the page, select the
notebook that you wish to view/run.

The notebook is divided into cells (some have text while other have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code
and produces it's output.
To execute the whole notebook, run all cells in order.


## Figures, tables and extra results not included in the paper

The figures and tables included in the paper are generated in the Jupyter notebooks,
and stored in the `figures/` and `latex_tables` folders, respectively.
Moreover, those folders also contain some extra figures and tables that are not 
included in the paper.


## License

All source code is made available under a MIT license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication.


