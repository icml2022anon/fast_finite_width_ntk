"""Minimal PyTorch NTK example."""
import torch
from fast_finite_width_ntk import empirical_pytorch

input_size = 8 * 8 * 3

x1 = torch.rand(6, input_size)
x2 = torch.rand(3, input_size)

# A vanilla FCN.
f = torch.nn.Sequential(
      torch.nn.Linear(input_size, 2048),
      torch.nn.ReLU(),
      torch.nn.Linear(2048, 2048),
      torch.nn.ReLU(),
      torch.nn.Linear(2048, 2048),
      torch.nn.ReLU(),
      torch.nn.Linear(2048, 10),
)

f.forward(x1)
params = list(f.parameters())

# TODO: figure out a proper way to match tree structures and tensor layouts for
#   Pytorch and respective ONNX models.
params = [p.T if p.ndim == 2 else p for p in params]

kwargs = dict(
    f=f,
    input_shape=x1.shape[1:],
    trace_axes=(),
    vmap_axes=0,
)


# Default, baseline Jacobian contraction.
jacobian_contraction = empirical_pytorch.empirical_ntk_fn_pytorch(
    **kwargs,
    implementation=empirical_pytorch.NtkImplementation.JACOBIAN_CONTRACTION)

# (6, 3, 10, 10) full `np.ndarray` test-train NTK
ntk_jc = jacobian_contraction(x2, x1, params)


# NTK-vector products-based implementation.
ntk_vector_products = empirical_pytorch.empirical_ntk_fn_pytorch(
    **kwargs,
    implementation=empirical_pytorch.NtkImplementation.NTK_VECTOR_PRODUCTS)


ntk_vp = ntk_vector_products(x2, x1, params)


# Structured derivatives-based implementation.
structured_derivatives = empirical_pytorch.empirical_ntk_fn_pytorch(
    **kwargs,
    implementation=empirical_pytorch.NtkImplementation.STRUCTURED_DERIVATIVES)

ntk_sd = structured_derivatives(x2, x1, params)


# Auto-FLOPs-selecting implementation. Doesn't work correctly on CPU/GPU.
auto = empirical_pytorch.empirical_ntk_fn_pytorch(
    **kwargs,
    implementation=empirical_pytorch.NtkImplementation.AUTO)

ntk_auto = auto(x2, x1, params)


# Check that implementations match
for ntk1 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
    for ntk2 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
        diff = torch.max(torch.abs(ntk1 - ntk2))
        print(f'NTK implementation diff {diff}.')
        assert diff < 1e-4

print('All NTK implementations match.')
