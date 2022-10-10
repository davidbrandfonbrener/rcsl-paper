# rcsl-paper

This is the codebase that accompanies our paper: [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079) which is to be published at [NeurIPS 2022](https://neurips.cc/)

The paper can be cited as:
```
@article{brandfonbrener2022does,
  title={When does return-conditioned supervised learning work for offline reinforcement learning?},
  author={Brandfonbrener, David and Bietti, Alberto and Buckman, Jacob and Laroche, Romain and Bruna, Joan},
  journal={arXiv preprint arXiv:2206.01079},
  year={2022}
}
```

## Python environment setup

```
conda env create --name rcsl-paper -f environment.yml
cd jax_continuous_rl
pip install -e .
```

## Codebase organization

All models and data generation code is in the `jax_continuous_rl/` directory except for decision transformers (DT) which we take from the original implementation and can be found in the `decision-transformer/` directory.

DT models are trained using `decision-transformer/gym/experiment.py`. All other models are trained using `jax_continuous_rl/experiments/train_offline.py` with sweeps defined by the files in `jax_continuous_rl/experiments/sweeps/`.


## Acknowledgements

Code for our implementations is based off of [jaxrl](https://github.com/ikostrikov/jaxrl) and [decision-transformer](https://github.com/kzl/decision-transformer).