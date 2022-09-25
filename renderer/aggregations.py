import jax.numpy as jnp
from jax import vmap

def aggregation_alpha(imgs,func):
    if func == 'probabilistic':
        img = imgs[0]
        for img_t in imgs[1:]:
            img = img_t + img - img_t * img
    elif func == 'maximum':
        img = jnp.amax(imgs,axis=0)
    elif func == 'average':
        img = jnp.where(imgs>0,imgs/imgs.sum(0),0)
    elif func == 'summation':
        img = imgs.sum(0)
    return img

def aggregation_rgb(imgs:'[n,h,w,3]',depth:'[n,h,w,1]')->'[n,h,w,3]':
    gamma = 1e-1 #[1e-1 ~ 1e4]
    exp_inv_depth = jnp.exp(depth/gamma)*imgs
    exp_inv_depth_sum = exp_inv_depth.sum(0)
    return jnp.where(exp_inv_depth_sum>0,exp_inv_depth/exp_inv_depth_sum,0)

batch_aggregation_rgb = vmap(aggregation_rgb,in_axes=(0,0),out_axes=(0))