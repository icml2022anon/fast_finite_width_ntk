# Fast Finite Width Neural Tangent Kernel has moved to https://github.com/google/neural-tangents
Anonymous code supplement for the ICML2022 submission "Fast Finite Width Neural Tangent Kernel". 

## JAX

To run `example.py`:
```bash
git clone https://github.com/icml2022anon/fast_finite_width_ntk.git 
python fast_finite_width_ntk/example.py
```

To run `example.ipynb`, you can open it in [Colab](https://colab.research.google.com/github/icml2022anon/fast_finite_width_ntk/blob/main/example.ipynb).

To use in other projects:
```commandline
pip install git+https://github.com/icml2022anon/fast_finite_width_ntk.git
```

## Tensorflow

In addition to the above, install [`tf2jax`](https://github.com/deepmind/tf2jax) without dependencies:
```commandline
pip install git+https://github.com/deepmind/tf2jax.git --no-deps
```

Then you can run `python example_tf.py` or open the `example_tf.ipynb` in [Colab](https://colab.research.google.com/github/icml2022anon/fast_finite_width_ntk/blob/main/example_tf.ipynb).

## Pytorch

Install `tf2jax` as above, and run `python example_pytorch.py` or open the `example_pytorch.ipynb` in [Colab](https://colab.research.google.com/github/icml2022anon/fast_finite_width_ntk/blob/main/example_pytorch.ipynb).
