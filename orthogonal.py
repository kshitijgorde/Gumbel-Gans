import tensorflow as tf
import numpy as np


def orthogonal_initializer(scale=1.0, seed=None, dtype=tf.float32):
    """Returns an initializer performing orthogonal initialization for weights.

    This function implements the weight initialization from:
    Andrew M. Saxe, James L. McClelland, Surya Ganguli (2014):
             Exact solutions to the nonlinear dynamics of learning in deep
             linear neural networks. International Conference on Learning
             Representations.
    This initializer is designed to reflect independent modes of variation
    in the input, by using a orthogonal projection of a random matrix.
    Args:
      scale: A Python float. Used to scale weights. Set to 1.0 if the weights
        will be input to a linear or sigmoid activation function. Use sqrt(2)
        for rectified linear units, and sqrt(2 / 1 + leakiness**2)) if leaky.
      seed: A Python integer. Used to create random seeds. See
        [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
        for behavior.
      dtype: The data type. Only floating point types are supported.
    Returns:
      An initializer for a weight matrix.
    """

    def _initializer(shape, dtype=dtype):
        flat = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = (u if u.shape == flat else v).reshape(shape)
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)

    return _initializer