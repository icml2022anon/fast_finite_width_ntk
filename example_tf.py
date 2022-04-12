"""Minimal Tensorflow NTK example."""
import tensorflow as tf
from tensorflow.keras import layers
from fast_finite_width_ntk import empirical_tf


x1 = tf.random.normal((6, 8, 8, 3))
x2 = tf.random.normal((3, 8, 8, 3))


def _check_ntks(f, params):
    kwargs = dict(
        f=f,
        trace_axes=(),
        vmap_axes=0,
    )
    # Default, baseline Jacobian contraction.
    jacobian_contraction = empirical_tf.empirical_ntk_fn_tf(
        **kwargs,
        implementation=empirical_tf.NtkImplementation.JACOBIAN_CONTRACTION)
    # (6, 3, 10, 10) full `np.ndarray` test-train NTK
    ntk_jc = jacobian_contraction(x2, x1, params)
    # NTK-vector products-based implementation.
    ntk_vector_products = empirical_tf.empirical_ntk_fn_tf(
        **kwargs,
        implementation=empirical_tf.NtkImplementation.NTK_VECTOR_PRODUCTS)
    ntk_vp = ntk_vector_products(x2, x1, params)
    # Structured derivatives-based implementation.
    structured_derivatives = empirical_tf.empirical_ntk_fn_tf(
        **kwargs,
        implementation=empirical_tf.NtkImplementation.STRUCTURED_DERIVATIVES)
    ntk_sd = structured_derivatives(x2, x1, params)
    # Auto-FLOPs-selecting implementation. Doesn't work correctly on CPU/GPU.
    auto = empirical_tf.empirical_ntk_fn_tf(
        **kwargs,
        implementation=empirical_tf.NtkImplementation.AUTO)
    ntk_auto = auto(x2, x1, params)

    # Check that implementations match
    for ntk1 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
        for ntk2 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
            diff = tf.reduce_max(tf.abs(ntk1 - ntk2))
            print(f'NTK implementation diff {diff}.')
            assert diff < 1e-4

    print('All NTK implementations match.')


# A vanilla CNN `tf.keras.Model` example.
print('A Keras CNN example.')

f = tf.keras.Sequential()
f.add(layers.Conv2D(32, (3, 3), activation='relu'))
f.add(layers.Conv2D(32, (3, 3), activation='relu'))
f.add(layers.Conv2D(32, (3, 3)))
f.add(layers.Flatten())
f.add(layers.Dense(10))

f.build((None, *x1.shape[1:]))
params = f.weights
_check_ntks(f, params)


# A `tf.function` example.
print('A `tf.function` example.')

params_tf = tf.random.normal((1, *x1.shape[1:]))


@tf.function(autograph=False,
             input_signature=[tf.TensorSpec(params_tf.shape),
                              tf.TensorSpec((None, *x1.shape[1:]))])
def f_tf(params, x):
    return x * tf.reduce_mean(params**2) + 1.


_check_ntks(f_tf, params_tf)
