# rcsl-paper

This is the codebase that accompanies our paper: [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/abs/2206.01079).

The paper can be cited as:
```
```

## Python environment setup

```
conda env create --name rcsl-paper -f environment.yml
cd jax_continuous_rl
pip install -e .
```

## Codebase organization

All models and data generation code is in the `jax_continuous_rl/` directory except for decision transformers which we take from the original implementation and can be found in the `decision-transformer/` directory.

Training sweeps are found in `jax_continuous_rl/experiments/`


## Acknowledgements

Code for our implementations is based off of [jaxrl]() and [decision-transformer]()