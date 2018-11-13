import numpy as np
def dot(u, v, geometry="spherical"):
    '''
        Calculate dot_product for two n-D vectors, u and v
        Inputs: u, v: two vectors, represented as np.arrays
        Outputs: dot_product, a 1-D real number
    '''
    if geometry == "spherical" or geometry == "euclidean":
        return u.dot(v)
    elif geometry == "hyperbolic":
        return u[:-1].dot(v[:-1])-u[-1]*v[-1]
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))

def project_to_tangent(point_on_manifold, displacement, geometry="spherical"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
    '''
    xp_dot = dot(point_on_manifold, displacement, geometry)
    xx_dot = dot(point_on_manifold, point_on_manifold, geometry)
#    print("xp_norm = {}, xx_norm = {}".format(xp_norm, xx_norm))
    return displacement - (xp_dot/xx_dot)*point_on_manifold

def exponential_map(v_tan, point_on_manifold, geometry="spherical"):
    '''
        Projects vector from tangent space of point_on_manifold onto manifold
        Inputs:
                v_tan is the n-D vector in tangent space, an np.array
                point_on_manifold is the initial n-D point on the manifold, an np.array
    '''
    norm_v_tan = np.sqrt(dot(v_tan, v_tan, geometry))
    print("norm_v_tan = ", norm_v_tan)
    if geometry == "spherical":
        return np.cos(norm_v_tan)*point_on_manifold + (np.sin(norm_v_tan)/norm_v_tan)*v_tan
    elif geometry == "hyperbolic":
        return np.cosh(norm_v_tan)*point_on_manifold + (np.sinh(norm_v_tan)/norm_v_tan)*v_tan
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))

def distance(u, v, geometry="spherical"):
    '''
        Calculate distance on the manifold between two pts
        Inputs: u, v: two vectors, represented as np.arrays
        Outputs: distance, a 1-D real number
    '''   
    dotprod = dot(u,v,geometry)
    if geometry == "spherical":
        return np.arccos(dotprod)
    elif geometry == "hyperbolic":
        return np.arccosh(-dotprod)
    elif geometry == "euclidean":
        return np.sqrt(dot(u-v, u-v, geometry))
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))