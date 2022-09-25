import jax.numpy as jnp
from jax.scipy.stats import norm
import jax
def get_kernels(sign,distance,blurriness,kernel):
    if kernel == 'hard':
        rasterise = (sign>0)
        # rasterise = jnp.where(sign>0,sign,0)
        
    elif kernel == 'gaussian':
        rasterise = norm.cdf(sign * distance / blurriness)

    elif kernel == 'logistic':
        # rasterise = 1. / (1. + jnp.exp(- sign * distance / blurriness))
        rasterise = jax.nn.sigmoid(sign * distance / blurriness)

    elif kernel == 'reciprocal':
        rasterise = sign * distance / blurriness / (1 + distance / blurriness) / 2. + 0.5

    elif kernel == 'cauchy':
        rasterise = jnp.arctan(sign * distance / blurriness) / jnp.pi + 0.5

    elif kernel == 'gudermannian':
        rasterise = jnp.arctan(jnp.tanh(sign * distance / blurriness / 2.)) * 2. / jnp.pi + 0.5

    elif kernel == 'gumbel_max':
        rasterise = jnp.exp(-jnp.exp(- sign * distance / blurriness))

    elif kernel == 'gumbel_min':
        rasterise = 1. - jnp.exp(-jnp.exp(- sign * distance / blurriness))

    else:
        assert False, "no kernel assigned"

    return rasterise