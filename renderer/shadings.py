import numpy as np
import jax.numpy as jnp
from util_general import normalize
import jax
def flat_shading(object,light,camera):
    view_points = object.mesh[:,:,:3].mean(1) # [n,3]
    normal = object.normal[:,:,:3].mean(1) # [n,3]
    n = normalize(normal,axis=-1)
    v = normalize(camera.position - view_points,axis=-1) # [n,3]
    l = normalize(light.position - view_points,axis=-1) # [n,3]
    h = normalize(l+v,axis=-1)
    r2 = jnp.linalg.norm(light.position - view_points,axis=-1)**2
    Ns = object.material['Ns']
    Ka = object.material['Ka']
    Kd = object.material['Kd']
    Ks = object.material['Ks']
    ambient = Ka[None,:]*light.intensity
    diffuse = Kd[None,:]*((light.intensity/r2)*jnp.fmax(0.,jnp.einsum('il,il->i',n,l)))[:,None]
    specular = Ks[None,:]*(((light.intensity/r2)*pow(jnp.fmax(0.,jnp.einsum('il,il->i',n,h)),Ns)))[:,None]
    return (ambient + diffuse + specular)[:,None,None,:] #[n,1,1,3]

def gourard_shading(object,light,camera):
    view_points = object.mesh[:,:,:3] # [n,3,3]
    normal = object.normal[:,:,:3] # [n,3,3]
    n = normalize(normal,axis=-1)
    v = normalize(camera.position - view_points,axis=-1) # [n,3,3]
    l = normalize(light.position - view_points,axis=-1) # [n,3,3]
    h = normalize(l+v,axis=-1)
    r2 = jnp.linalg.norm(light.position - view_points,axis=-1)**2
    Ns = object.material['Ns']
    Ka = object.material['Ka']
    Kd = object.material['Kd']
    Ks = object.material['Ks']
    ambient = Ka[None,None,:]*light.intensity
    diffuse = Kd[None,None,:]*((light.intensity/r2)*jnp.fmax(0.,jnp.einsum('ijk,ijk->ij',n,l)))[:,:,None]
    specular = Ks[None,None,:]*(((light.intensity/r2)*pow(jnp.fmax(0.,jnp.einsum('ijk,ijk->ij',n,h)),Ns)))[:,:,None]
    color = ambient + diffuse + specular #[n,3,3]
    return jax.lax.stop_gradient(object.depth)*(color[:,:,None,None,:]*jax.lax.stop_gradient(object.depth_inv)).sum(axis=1) #[n,h,w,3]


def phong_shading(object,light,camera):
    normal = jax.lax.stop_gradient(object.depth)*(object.normal[:,:,None,None,:]*jax.lax.stop_gradient(object.depth_inv)).sum(axis=1) #[n,h,w,3]
    view_points_xy = jax.lax.stop_gradient(object.depth)*(object.mesh[:,:,:2][:,:,None,None,:]*jax.lax.stop_gradient(object.depth_inv)).sum(axis=1) 
    view_points = jnp.concatenate([view_points_xy,jax.lax.stop_gradient(object.depth)],axis=-1) #[n,h,w,3]
    n = normalize(normal,axis=-1)
    v = normalize(camera.position - view_points,axis=-1) # [n,h,w,3]
    l = normalize(light.position - view_points,axis=-1) # [n,h,w,3]
    h = normalize(l+v,axis=-1)
    r2 = jnp.linalg.norm(light.position - view_points,axis=-1)**2 #[n,h,w]
    Ns = object.material['Ns']
    Ka = object.material['Ka']
    Kd = object.material['Kd']
    Ks = object.material['Ks']
    ambient = Ka[None,None,None,:]*light.intensity
    diffuse = Kd[None,None,None,:]*((light.intensity/r2)*jnp.fmax(0.,jnp.einsum('ijkl,ijkl->ijk',n,l)))[:,:,:,None]
    specular = Ks[None,None,None,:]*(((light.intensity/r2)*pow(jnp.fmax(0.,jnp.einsum('ijkl,ijkl->ijk',n,h)),Ns)))[:,:,:,None]
    return ambient + diffuse + specular