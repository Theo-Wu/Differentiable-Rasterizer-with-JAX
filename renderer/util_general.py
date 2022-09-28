import jax.numpy as jnp
import jax
from functools import partial
from kernels import get_kernels

@partial(jax.jit, static_argnums=(1,))
def normalize(v,axis=-1):
    safe_norm = jnp.where(jnp.linalg.norm(v,axis=axis)!=0,jnp.linalg.norm(v,axis=axis),1e-6)
    return v/safe_norm[...,None]

def p2f(points,polygon_position):
    vertexCount = polygon_position.shape[1] #[n,t,3]
    min_dis = 1e6
    sign = 1.
    for i in range(vertexCount):
        f = edge(points,polygon_position[:,i%vertexCount],polygon_position[:,(i+1)%vertexCount])
        sign *= jnp.where(f>0,1.,0.) 
        p1 = polygon_position[:,i%vertexCount,0:2]
        p2 = polygon_position[:,(i+1)%vertexCount,0:2]
        dis = point_to_line_distance_parallel(points,p1,p2)
        min_dis = jnp.fmin(min_dis,dis)
    sign = 2*sign-1. #inside positive outside negative
    return min_dis,sign
    
def edge(p,a,b)->'negative value for anticlockwise':
    h,w,_ = p.shape
    c,_ = a.shape
    px = p[:,:,0]
    py = p[:,:,1]
    ax = a[:,0]
    ay = a[:,1]
    bx = b[:,0]
    by = b[:,1]
    f = px*jnp.broadcast_to((by-ay)[...,None,None],(c,h,w)) - py*jnp.broadcast_to((bx-ax)[...,None,None],(c,h,w)) + jnp.broadcast_to((ay*bx-ax*by)[...,None,None],(c,h,w))
    return -f #[n,h,w]

def getArea(c,a,b):
    cx = c[:,0]
    cy = c[:,1]
    ax = a[:,0]
    ay = a[:,1]
    bx = b[:,0]
    by = b[:,1]
    f = cx*(by-ay) - cy*(bx-ax) + (ay*bx-ax*by)
    return -f #[n,]

def getArea_n(polygon_position):
    n,t,_ = polygon_position.shape
    area = 0
    for i in range(1,t-1):
        area += getArea(polygon_position[:,0],polygon_position[:,i],polygon_position[:,i+1])
    return area #[n,]

def getnormal(mesh):
    if len(mesh.shape)==3: #[n,3,3]
        BA = mesh[:,1,:3]-mesh[:,0,:3]
        CA = mesh[:,2,:3]-mesh[:,0,:3]
        normal = jnp.cross(BA,CA) #[n,3]
    elif len(mesh.shape)==4: #[b,n,3,3]
        BA = mesh[:,:,1,:3]-mesh[:,:,0,:3]
        CA = mesh[:,:,2,:3]-mesh[:,:,0,:3]
        normal = jnp.cross(BA,CA) #[b,n,3]
    return normalize(normal,-1)

def getUV(points,polygon_position):
    area = getArea(polygon_position[:,0],polygon_position[:,1],polygon_position[:,2])
    w0 = edge(points,polygon_position[:,1],polygon_position[:,2])/area[:,None,None]
    w1 = edge(points,polygon_position[:,2],polygon_position[:,0])/area[:,None,None]
    w2 = edge(points,polygon_position[:,0],polygon_position[:,1])/area[:,None,None]
    e0 = w0.at[(w0<0.) | (w1<0.) | (w2<0.)].set(1./3.)
    e1 = w1.at[(w0<0.) | (w1<0.) | (w2<0.)].set(1./3.)
    e2 = w2.at[(w0<0.) | (w1<0.) | (w2<0.)].set(1./3.)
    # e0 = jnp.where((w0<0.) | (w1<0.) | (w2<0.),0,w0)
    # e1 = jnp.where((w0<0.) | (w1<0.) | (w2<0.),0,w1)
    # e2 = jnp.where((w0<0.) | (w1<0.) | (w2<0.),0,w2)
    return jnp.stack([e0,e1,e2],axis=1) #[n,3,h,w]

def project(polygon_positions,camera,viewport,zNear,zFar):
    target_point = camera.lookat
    view_point = camera.position
    view_up = camera.up
    view_plane_normal = target_point-view_point
    n = -(view_plane_normal)/jnp.linalg.norm(view_plane_normal) # front look at negative z
    u = (jnp.cross(view_up,n))/jnp.linalg.norm(jnp.cross(view_up,n)) # left hand
    v = (jnp.cross(n,u))/jnp.linalg.norm(jnp.cross(n,u)) # actual up
    viewMatrix = jnp.array([[u[0],u[1],u[2],-jnp.dot(view_point,u)],
                            [v[0],v[1],v[2],-jnp.dot(view_point,v)],
                            [n[0],n[1],n[2],-jnp.dot(view_point,n)],
                            [0.,0.,0.,1.0]])

    aspect = viewport[0]/viewport[1]
    l = -1.0
    r = 1.0
    b = -1.0
    t = 1.0

    projectMatrix = jnp.array([[2*zNear/(r-l),0.,-(l+r)/(r-l),0.],
                                [0., 2*zNear/(t-b),-(t+b)/(t-b),0.],
                                [0.,0.,(zNear+zFar)/(zNear-zFar),-2*zNear*zFar/(zNear-zFar)],
                                [0.,0.,1.,0.]])

    mvp = (projectMatrix@viewMatrix@polygon_positions.transpose(0,2,1)).transpose(0,2,1)
    projected_polygon_positions = mvp.at[...,0:-1].set(mvp[...,0:-1]/mvp[...,-1][:,:,None])

    return projected_polygon_positions

def get_points_from_angles(elevation, azimuth, distance, degrees=True):
    if isinstance(distance, float) or isinstance(distance, int):
        if degrees:
            elevation = jnp.radians(elevation)
            azimuth = jnp.radians(azimuth)
        return jnp.array([
            distance * jnp.cos(elevation) * jnp.cos(azimuth),
            distance * jnp.sin(elevation),
            distance * jnp.cos(elevation) * jnp.sin(azimuth)])
    else:
        if degrees:
            elevation = jnp.pi / 180. * elevation
            azimuth = jnp.pi / 180. * azimuth
    #
        return jnp.stack([
            distance * jnp.cos(elevation) * jnp.cos(azimuth),
            distance * jnp.sin(elevation),
            distance * jnp.cos(elevation) * jnp.sin(azimuth)
            ]).transpose(1, 0)

def get_angles_from_points(point, degrees=True):
    x,y,z = point
    distance = jnp.linalg.norm(point)
    elevation = jnp.arcsin(y/distance)
    azimuth = jnp.arcsin(z/(distance*jnp.cos(elevation)))
    if degrees:
        elevation = jnp.degrees(elevation)
        azimuth = jnp.degrees(azimuth)
    return jnp.array([elevation,azimuth,distance])

def rasterize(sign:'needed for euc distance',
                distance:'value of euc or edge',
                kernel:'kernel distribution name',
                blurriness:'1e-n'=1e-7
                ):
    rasterise_value = get_kernels(sign,distance,blurriness,kernel)
    return rasterise_value #[b,h,w]

def compute_line_magnitude(x1, y1, x2, y2):
    lineMagnitude = jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return lineMagnitude

def point_to_line_distance_parallel(points, p1, p2):
    px = points[:,:,0]
    py = points[:,:,1]
    x1 = p1[:,0]
    y1 = p1[:,1]
    x2 = p2[:,0]
    y2 = p2[:,1]
    num_tri = len(x1)

    x1 = x1[:,None,None]
    y1 = y1[:,None,None]
    x2 = x2[:,None,None]
    y2 = y2[:,None,None]
    px = jnp.tile(px,(num_tri,1,1))
    py = jnp.tile(py,(num_tri,1,1))

    line_magnitude = compute_line_magnitude(x1, y1, x2, y2)

    # Don't need to consider segment is too short

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))

    u = u1 / (line_magnitude * line_magnitude)

    # The projection of the point is inside the line segment
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    distance = compute_line_magnitude(px, py, ix, iy)

    # If the projection of the point to the line is not in the line segment
    d1 = compute_line_magnitude(px, py, x1, y1)
    d2 = compute_line_magnitude(px, py, x2, y2)
    mask = ((u<1e-5) + (u>1.))
    out_distance = jnp.fmin(d1,d2) * mask
    in_distance = distance * (1. - mask)
    distance = in_distance + out_distance

    return distance 