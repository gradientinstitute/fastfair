# Fast Fair Regression

This package provides the regression models and measures found in the papers,

[Steinberg, D., Reid, A., O'Callaghan, S., Lattimore, F., McCalman, L., & Caetano, T. (2020). Fast fair regression via efficient approximations of mutual information. arXiv preprint arXiv:2002.06200.](https://arxiv.org/pdf/2002.06200.pdf)

and

[Steinberg, D., Reid, A., & O'Callaghan, S. (2020). Fairness measures for regression via probabilistic classification. arXiv preprint arXiv:2001.06089.](https://arxiv.org/pdf/2001.06089.pdf)

We only provide the minimal code required to implement these models and
measures. We have chosen not to provide our complete experimental pipeline
as it depends on code we do not own, and do not have permission to redistribute.

## Installation

We use [python poetry](https://python-poetry.org/) for convenience, once
you have installed this package for your system, issue the following commands,

    git clone git@github.com:gradientinstitute/fastfair.git
    poetry install
    mkdir data
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data --output data/communities.data
    curl https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names --output data/communities.names
    poetry shell

You should now be ready to use the models and measures.

## Running the code

We demonstrate the models and measures in two Jupyter notebooks.
- `notebooks/measures.ipynb` demonstrates the use of probabilistic 
  classification for measuring fairness in a regression setting. It also 
  reproduces some plots from the "Fairness measures for regression via probabilistic classification" paper. 
- `notebooks/regression.ipynb` demonstrates the use of the fast fair regression
  models on the "Communities and crime" dataset (which is downloaded in the 
  installation instructions). Each of the fairness models are run with different
  fairness regularisation strengths to explore how this effects predictive and 
  fairness performance.
  
To run these notebooks simply type the commands,

    cd notebooks
    jupyter notebook regression.ipynb
  
Or substitute `regression.ipynb` with `measures.ipynb` on the last line.

## License

We provide this code under the Apache 2.0 license. See the `LICENSE` file for 
more information.

Copyright 2023 Gradient Institute