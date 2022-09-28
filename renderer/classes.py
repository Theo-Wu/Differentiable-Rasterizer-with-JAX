import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from util_general import *
from aggregations import *
from shadings import *

@register_pytree_node_class
class Object:
    def __init__(self,mesh,normal) -> None:
        self.mesh = mesh
        self.normal = normal
        
    def cal_project(self,camera,viewport,znear,zfar):
        self.projected_mesh = project(self.mesh,camera,viewport,znear,zfar)

    def cal_depth(self,points):
        self.bary_w = getUV(points,self.projected_mesh) #[n,3,h,w]
        self.depth_inv = (self.bary_w/self.projected_mesh[:,:,2][:,:,None,None])[...,None] #[n,3,h,w,1]

        self.depth = self.depth_inv.sum(axis=1) #[n,h,w,1]
        self.depth = jnp.where(self.depth!=0,1./self.depth,0.)

    def cal_distance(self,points):
        self.distance,self.sign = p2f(points,self.projected_mesh)

    def tree_flatten(self):
        children = (self.mesh,self.normal,)  # arrays / dynamic values
        aux_data = None
                    # {'projected_mesh':self.projected_mesh,
                    # 'distance':self.distance,
                    # 'sign':self.sign,
                    # 'depth':self.depth,
                    # 'depth_inv':self.depth_inv,
                    # 'bary_w':self.bary_w
                    # }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class Material:
    def __init__(self,K,Ns) -> None:
        self.K = K
        self.Ns = Ns

    def tree_flatten(self):
        children = (self.K,self.Ns)  # arrays / dynamic values
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class Light:
    def __init__(self,position,intensity) -> None:
        self.position = position
        self.intensity = intensity
    
    def tree_flatten(self):
        children = (self.position,self.intensity)  # arrays / dynamic values
        aux_data = None  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class Camera:
    def __init__(self,position,up,lookat) -> None:
        self.position = position
        self.up = up
        self.lookat = lookat
    
    def tree_flatten(self):
        children = (self.position,self.up,self.lookat)  # arrays / dynamic values
        # aux_data = {'up':self.up,'lookat':self.lookat}  # static values
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@register_pytree_node_class
class Scene:
    def __init__(self,Light,Camera) -> None:
        self.Light = Light
        self.Camera = Camera
        self.Object = []
    
    def addObject(self,obj):
        self.Object.append(obj)

    def tree_flatten(self):
        children = (self.Light,self.Camera)  # arrays / dynamic values
        aux_data = None  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

def shading(object,material,light,camera,shading_mode):
    if shading_mode == 'flat':
        return flat_shading(object,material,light,camera)
    elif shading_mode == 'gourard':
        return gourard_shading(object,material,light,camera)
    elif shading_mode == 'phong':
        return phong_shading(object,material,light,camera)
    
def forward(points,objects,materials,light,camera,shading_mode,kernel,blurriness,aggregate_rgb=True):
    viewport = points.shape[0:2]
    blurriness = pow(10,blurriness)
    # imgs_scene = []
    # depth_scene = []
    for i in range(len(objects)):
        object = objects[i]
        material = materials[i]
        object.cal_project(camera,viewport,-1.,-100.)
        object.cal_depth(points)
        object.cal_distance(points)
        imgs = rasterize(object.sign,object.distance,kernel,blurriness)[...,None] #[n,h,w,1]
        # color
        if aggregate_rgb:
            imgs = aggregation_rgb(imgs,object.depth)
        imgs = imgs * shading(object,material,light,camera,shading_mode)
 
        # imgs_scene = jnp.concatenate([imgs_scene,imgs],axis=0) if len(imgs_scene) else imgs #[n1+n2+...,h,w,3]
        # depth_scene = jnp.concatenate([depth_scene,object.depth],axis=0) if len(depth_scene) else object.depth #[n1+n2+...,h,w,1]

    img = aggregation_alpha(imgs,'probabilistic')

    return img
