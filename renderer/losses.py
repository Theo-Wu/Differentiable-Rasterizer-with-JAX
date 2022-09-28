from classes import forward
import jax
import jax.numpy as jnp
def mse_loss(points,objects,materials,light,camera,img_ref,shading_mode,kernel,blurriness,aggregate_rgb=True):
    predict = forward(points,objects,materials,light,camera,shading_mode,kernel,blurriness,aggregate_rgb)
    dimg = jnp.sum((predict - img_ref)**2)
    return dimg, predict

# def mse_loss(points,objects,light,camera,camera2,shading_mode,kernel,blurriness):
#     predict = forward(points,objects,light,camera,shading_mode,kernel,blurriness)
#     img_ref = forward(points,objects,light,camera2,shading_mode,'hard',blurriness)
#     dimg = jnp.sum((predict - jax.lax.stop_gradient(img_ref))**2)
#     return dimg

# def iou_loss(points,objects,light,camera,camera2,shading_mode,kernel,blurriness):
#     predict = forward(points,objects,light,camera,shading_mode,kernel,blurriness)
#     img_ref = forward(points,objects,light,camera2,shading_mode,'hard',blurriness)
#     intersect = (predict * jax.lax.stop_gradient(img_ref))
#     union = (predict + jax.lax.stop_gradient(img_ref) - predict * jax.lax.stop_gradient(img_ref)) + 1e-6
#     return (1. - intersect / union).sum()

# def mix_loss(points,objects,light,camera,camera2,shading_mode,kernel,blurriness):
#     predict = forward(points,objects,light,camera,shading_mode,kernel,blurriness)
#     img_ref = forward(points,objects,light,camera2,shading_mode,'hard',blurriness)
#     intersect = (predict * jax.lax.stop_gradient(img_ref))
#     union = (predict + jax.lax.stop_gradient(img_ref) - predict * jax.lax.stop_gradient(img_ref)) + 1e-6
#     iou_loss = (1. - intersect / union).sum()
#     mse_loss = ((predict - jax.lax.stop_gradient(img_ref))**2).sum()
#     return 0.5*mse_loss + 0.5*iou_loss