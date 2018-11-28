import numpy as np
def dot(u, v, geometry="spherical"):
    '''
        Calculate dot_product for two sets of m n-dimensional vectors, u and v
        Inputs: u, v: two arrays containing m n-dim vectors, as columns or rows
        Outputs: a 1-D array of real numbers corresponding to the dot products
    '''
    metric = get_metric(u.shape[0], geometry)
    return np.squeeze(np.diag(u.T.dot(metric.dot(v))))  
 

def project_to_tangent_old(point_on_manifold, displacement, geometry="spherical"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
        Outputs: n-D vector in tangent space
        NOTE: This explicitly calculates <x,x>, which is somewhat wasteful and a potential
            source of inaccuracy, since <x, x> = 1 if x is on spherical manifold, or =-1 on
            hyperboloid
    '''
    print("project_to_tangent: point_on_manifold = {}, displacement = {}, geometry = {}".format(
            point_on_manifold, 
            displacement,
            geometry
           )
         )

    xp_dot = dot(point_on_manifold, displacement, geometry)
    xx_dot = dot(point_on_manifold, point_on_manifold, geometry)
#    print("project_to_tangent: xp_norm = {}, xx_norm = {}".format(xp_dot, xx_dot))
    return displacement - (xp_dot/xx_dot)*point_on_manifold

def project_to_tangent(point_on_manifold, displacement, geometry="spherical"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
    '''
    print("project_to_tangent: point_on_manifold = {}, displacement = {}, geometry = {}".format(
            point_on_manifold, 
            displacement,
            geometry
           )
         )

    xp_dot = dot(point_on_manifold, displacement, geometry)
    xx_dot = +1. #if on spherical manifold
    if geometry in "hyperbolic":
        xx_dot = -1. #if on hyperboloid manifold
#    print("project_to_tangent: xp_norm = {}, xx_norm = {}".format(xp_dot, xx_dot))
    return displacement - (xp_dot/xx_dot)*point_on_manifold

        
def exponential_map(v_tan, point_on_manifold, geometry="spherical"):
    '''
        Projects vector from tangent space of point_on_manifold onto manifold
        Inputs:
                v_tan is the n-D vector in tangent space, an np.array
                point_on_manifold is the initial n-D point on the manifold, an np.array
    '''
    norm_v_tan = np.sqrt(dot(v_tan, v_tan, geometry))
#    print("exponential_map: norm_v_tan = ", norm_v_tan)
    if abs(norm_v_tan) < 1e-8:
        return point_on_manifold
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
        
def get_metric(dimension, geometry="euclidean"):
    '''
        Form metric for various geometries.
        Inputs: dimension: an integer specifying size of square metric matrix
        Outputs: (d x d) np.array containing the metric terms
    '''
    metric = np.eye(dimension)
    if geometry == "hyperbolic":        
        metric[-1, -1] = -1.
    elif geometry == "euclidean":
        pass
    elif geometry == "spherical":
        #This probably needs to be renamed, since it's not in terms of r, θ, φ
        pass
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))
        return np.zeros([dimension, dimension])

    return metric
