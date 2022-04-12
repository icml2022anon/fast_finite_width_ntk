"""Fast computation of empirical NTK in PyTorch.

All functions in this module are applicable to any `torch.nn.Module`, subject
to compatibility with `onnx` and `tf2jax`.

The NTK kernels have a very specific output shape convention that may be
unexpected. Further, NTK has multiple implementations that may perform
differently depending on the task.
Please read individual functions' docstrings.

Example:
  >>>  import torch
  >>>  from fast_finite_width_ntk import empirical_ntk_fn_pytorch
  >>>
  >>>  x_train = torch.rand((20, 32 * 32 * 3))
  >>>  x_test = torch.rand(5, 32 * 32 * 3))
  >>>
  >>>  # An FCN
  >>>  f = torch.nn.Sequential(
  >>>      torch.nn.Linear(x_train.shape[1], 2048),
  >>>      torch.nn.ReLU(),
  >>>      torch.nn.Linear(2048, 2048),
  >>>      torch.nn.ReLU(),
  >>>      torch.nn.Linear(2048, 2048),
  >>>      torch.nn.ReLU(),
  >>>      torch.nn.Linear(2048, 10),
  >>>  )
  >>>
  >>>  f.forward(x_train)
  >>>  params = [p.T if p.ndim == 2 else p for p in f.parameters()]
  >>>
  >>>  # Use ntk-vector products since the network has many parameters
  >>>  # relative to the cost of forward pass.
  >>>  ntk_fn = empirical_ntk_fn(f, vmap_axes=0, implementation=2)
  >>>
  >>>  # (5, 5) np.ndarray test-test NTK
  >>>  ntk_test_test = ntk_fn(x_test, None, params)
  >>>
  >>>  # Compute only NTK diagonal variances:
  >>>  ntk_fn = empirical_ntk_fn(f, diagonal_axes=(0,))
  >>>
  >>>  # (20,) np.ndarray train-train NTK diagonal
  >>>  ntk_train_train_diag = ntk_fn(x_train, None, params)
"""

import warnings
from typing import Union, Callable, Optional, Tuple
import tensorflow as tf
import tf2jax
import jax.dlpack
import torch
import onnx
from onnx2keras import onnx_to_keras
from fast_finite_width_ntk.utils.typing import NTTree, PyTree, Axes, VMapAxes
from fast_finite_width_ntk.empirical import empirical_ntk_fn, NtkImplementation


def empirical_ntk_fn_pytorch(
    f: torch.nn.Module,
    input_shape: Tuple[int, ...],
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = NtkImplementation.STRUCTURED_DERIVATIVES,
    fwd: Optional[bool] = None,
    j_rules: bool = True,
    a_rules: bool = True,
    change_ordering: bool = False
) -> Callable[[NTTree[torch.Tensor],
               Optional[NTTree[torch.Tensor]],
               PyTree],
              NTTree[torch.Tensor]]:
  r"""Returns a function to draw a single sample the NTK of a given network `f`.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the Jacobian :math:`df/dparams` of shape
  `full_output_shape + params.shape`.

  For best performance:
  1) pass `x2=None` if `x1 == x2;
  2) prefer square batches (i.e `x1.shape == x2.shape`);
  3) make sure to set `vmap_axes` correctly.
  4) try different `implementation` values.

  WARNING: Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
  subject to `trace_axes` and `diagonal_axes` parameters, which make certain
  assumptions about the outputs `f(x)` that may only be true in the infinite
  width / infinite number of samples limit, or may not apply to your
  architecture. For most precise results in the context of linearized training
  dynamics of a specific finite-width network, set both `trace_axes=()` and
  `diagonal_axes=()` to obtain the kernel exactly of shape
  `zip(f(x1).shape, f(x2).shape)`.

  For networks with multiple (i.e. lists, tuples, PyTrees) outputs, in principal
  the empirical kernels will have terms measuring the covariance between the
  outputs. Here, we ignore these cross-terms and consider each output
  separately. Please raise an issue if this feature is important to you.

  Args:
    f:
      the Pytorch `torch.nn.Module` whose NTK we are computing.

    input_shape:
      input shape of a single input to the model, needed for shape inference.

    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

    vmap_axes:
      A triple of `(in_axes, out_axes, kwargs_axes)`
      passed to `vmap` to evaluate the empirical NTK in parallel ove these axes.
      Precisely, providing this argument implies that `f.call(x, **kwargs)`
      equals to a concatenation along `out_axes` of `f` applied to slices of
      `x` and `**kwargs` along `in_axes` and `kwargs_axes`. In other words, it
      certifies that `f` can be evaluated as a `vmap` with `out_axes=out_axes`
      over `x` (along `in_axes`) and those arguments in `**kwargs` that are
      present in `kwargs_axes.keys()` (along `kwargs_axes.values()`).

      This allows us to evaluate Jacobians much more
      efficiently. If `vmap_axes` is not a triple, it is interpreted as
      `in_axes = out_axes = vmap_axes, kwargs_axes = {}`. For example a very
      common use case is `vmap_axes=0` for a neural network with leading (`0`)
      batch dimension, both for inputs and outputs, and no interactions between
      different elements of the batch (e.g. no BatchNorm, and, in the case of
      `nt.stax`, also no Dropout). However, if there is interaction between
      batch elements or no concept of a batch axis at all, `vmap_axes` must be
      set to `None`, to avoid wrong (and potentially silent) results.

    implementation:
      `1`, `2`, `3`, or `0`.

      `0` selects the best of `1`, `2`, and `3` based on FLOPs analysis.
      It only works correctly for TPUs, and on CPU/GPU returns wrong FLOPs and
      may select a slower method.

      `1` directly instantiates Jacobians and computes their contraction.

      `2` uses NTK-vector products to avoid expensive contraction at the
      cost of extra forward and backward passes through the network.

      `3` uses structured derivatives to simplify the NTK contraction.

    j_rules:
      `True` to allow custom Jacobian rules for `dy/dw` computations.

    a_rules:
      `True` to allow simplification rules for structured `dy/dw` derivatives.

    fwd:
      `True` to allow `jvp` in intermediary kernel computations, `False` to
      always use `vjp`. `None` to decide based on input/output sizes.

    change_ordering:
      argument to switch `CHW` to `HWC` channel order in CNNs when converting
      Pytorch models to TF/JAX. TODO: make Pytorch and TF/JAX params properly
      match tree structures and tensor layouts for ONNX CNNs.

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  warnings.warn('This function is an early proof-of-concept.')

  kwargs = dict(
      trace_axes=trace_axes,
      diagonal_axes=diagonal_axes,
      vmap_axes=vmap_axes,
      fwd=fwd,
      j_rules=j_rules,
      a_rules=a_rules,
      implementation=implementation
  )

  x = torch.randn((1, *input_shape), requires_grad=True)
  f.forward(x)
  torch.onnx.export(f, x, 'f.onnx',
                    input_names=['input'], output_names=['output'])
  onnx_model = onnx.load('f.onnx')
  f_tf = onnx_to_keras(onnx_model, ['input'], change_ordering=change_ordering)

  if isinstance(f_tf, tf.keras.Model):

    @tf.function
    def forward_tf(x):
      return f_tf.call(x, training=False)

    if change_ordering:
      input_shape_tf = (input_shape[-1], *input_shape[1:-1], input_shape[0])
    else:
      input_shape_tf = input_shape

    apply_fn_, params_tf = tf2jax.convert(
      forward_tf,
      tf.TensorSpec((None, *input_shape_tf)))

    params_keys = list(params_tf.keys())

    def apply_fn(params, x):
      assert len(params) == len(params_keys), (len(params), params_keys)
      params = {key: p for key, p in zip(params_keys, params)}
      return apply_fn_(params, x)[0]

  else:
    raise TypeError(type(f_tf), f_tf)

  jax_ntk_fn = empirical_ntk_fn(apply_fn, **kwargs)

  def ntk_fn(x1: NTTree[torch.Tensor],
             x2: Optional[NTTree[torch.Tensor]],
             params: PyTree,
             **apply_fn_kwargs) -> torch.Tensor:
    x1 = _torch_to_jax(x1)
    x2 = _torch_to_jax(x2)
    params = _torch_to_jax(params)
    out = jax_ntk_fn(x1, x2, params, **apply_fn_kwargs)
    return _jax_to_torch(out)

  return ntk_fn


def _torch_to_jax(x):
  return jax.tree_map(lambda x: jax.dlpack.from_dlpack(torch.to_dlpack(x)), x)


def _jax_to_torch(x):
  return jax.tree_map(lambda x: torch.from_dlpack(jax.dlpack.to_dlpack(x)), x)
