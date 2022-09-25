import numpy as np
from util_general import normalize

def load_obj(filename_obj, normalization=False, texture_type='surface'):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    assert texture_type in ['surface', 'vertex']

    with open(filename_obj) as f:
        lines = f.readlines()

    vertices = []
    faces = []
    vertex_normals = []

    for line in lines:
        if len(line.split()) == 0:
            continue
            
        # load vertices
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
                
        # load normal
        if line.split()[0] == 'vn':
            vertex_normals.append([float(v) for v in line.split()[1:4]])
     
    vertices = np.vstack(vertices).astype(np.float32)
    vertex_normals = np.vstack(vertex_normals).astype(np.float32)
    
    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2
        
        vertex_normals -= vertex_normals.min(0)[None, :]
        vertex_normals /= np.abs(vertex_normals).max()
        vertex_normals *= 2
        vertex_normals -= vertex_normals.max(0)[None, :] / 2

    # average the normal
    normals_aver = np.zeros_like(vertices)
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])-1
            n0 = int(vs[0].split('/')[-1])-1
            normals_aver[v0] += vertex_normals[n0]
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])-1 
                v2 = int(vs[i + 2].split('/')[0])-1
                faces.append([v0, v1, v2])
                n1 = int(vs[i+1].split('/')[-1])-1
                n2 = int(vs[i+2].split('/')[-1])-1
                normals_aver[v1] += vertex_normals[n1]
                normals_aver[v2] += vertex_normals[n2]

    faces = np.vstack(faces).astype(np.int32)
    normals_aver = normalize(normals_aver,-1)
    return vertices, faces, vertex_normals, normals_aver

def load_mtl(filename_mtl):
    '''
    load color (Kd) and filename of textures from *.mtl
    '''
    texture_filenames = {}
    material = {}
    material_name = ''
    with open(filename_mtl) as f:
        for line in f.readlines():
            if len(line.split()) != 0:
                if line.split()[0] == 'newmtl':
                    material_name = line.split()[1]
                    material[material_name] = np.zeros([3,3])
                    
                if line.split()[0] == 'map_Ka':
                    texture_filenames[material_name] = line.split()[1]
                if line.split()[0] == 'Ka':
                    material[material_name,'Ka'] = np.array(list(map(float, line.split()[1:4])))

                if line.split()[0] == 'Kd':
                    material[material_name,'Kd'] = np.array(list(map(float, line.split()[1:4])))

                if line.split()[0] == 'Ks':
                    material[material_name,'Ks'] = np.array(list(map(float, line.split()[1:4])))
                    
    return material, texture_filenames
