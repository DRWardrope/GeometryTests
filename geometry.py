import numpy as np
def dot(u, v, geometry="spherical"):
    '''
        Calculate dot_product for two n-D vectors, u and v
        Inputs: u, v: two vectors, represented as np.arrays
        Outputs: dot_product, a 1-D real number
    '''
    metric = get_metric(u.shape[0], geometry)
    #print("u.T.shape = {}, v.shape = {}, metric.shape = {}".format(
    #            u.T.shape, v.shape, metric.shape,
    #        )
    #    )
    return u.T.dot(metric.dot(v))
 

def project_to_tangent_slow(point_on_manifold, displacement, geometry="spherical"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
        Outputs: n-D vector in tangent space
        NOTE: This explicitly calculates <x,x>, which is somewhat wasteful and a potential
            source of inaccuracy, but it does ensure that any geometry can be considered.
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
    return displacement - np.diag(xp_dot/xx_dot)*point_on_manifold

def project_to_tangent(point_on_manifold, displacement, geometry="hyperbolic"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
        NOTE: Uses pre-calculated values for <x,x>, only appropriate for Euclidean,
               spherical and hyperboloid geometries.
    '''
    print("project_to_tangent: point_on_manifold = {}, displacement = {}, geometry = {}".format(
            point_on_manifold,
            displacement,
            geometry
           )
         )

    xp_dot = np.diag(dot(point_on_manifold, displacement, geometry))
    xx_dot = np.ones(xp_dot.shape)
    if geometry in "hyperbolic":
        xx_dot *= -1. #if on hyperboloid manifold
    return displacement - (xp_dot/xx_dot)*point_on_manifold
        
def exponential_map(point_on_manifold, v_TpS, geometry="spherical"):
    '''
        Projects vector from tangent space of point_on_manifold onto manifold
        Inputs:
                point_on_manifold is a tf.Tensor, the initial n-D point, or 
                an array of n-D points on the manifold, 
                v_TpS is a tf.Tensor, the n-D vector, or array of such vectors,
                in tangent space
    '''
    norm_v_TpS = np.diag(np.sqrt(dot(v_TpS, v_TpS, geometry)))
       
    if geometry == "spherical":
        #if abs(norm_v_TpS.squeeze()) < 1e-8:
        #    return point_on_manifold
        #mapped_pt = tf.cond(norm_v_TpS.squeeze()) < 1e-8
        return np.cos(norm_v_TpS)*point_on_manifold + (np.sin(norm_v_TpS)/norm_v_TpS)*v_TpS
    elif geometry == "hyperbolic":
    #    print(norm_v_TpS)
    #    print(np.where(np.greater(norm_v_TpS , 0.), "Y", "N"))
        return np.where(
                        np.greater(norm_v_TpS , 0.),
                        np.cosh(norm_v_TpS)*point_on_manifold 
                            + (np.sinh(norm_v_TpS)/norm_v_TpS)*v_TpS,
                        point_on_manifold
        )
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))

def distance(u, v, geometry="spherical"):
    '''
        Calculate distance on the manifold between two pts
        Inputs: u, v: two vectors, represented as np.arrays
        Outputs: distance, a 1-D real number
    '''   
    dotprod = dot(u,v,geometry)
#    if np.abs(dotprod) > 1:
#        print("distance: {}.{} = {:.3g}".format(u, v, dotprod))
    
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

def project_to_klein(v):
    '''
        Project hyperboloid points to Beltrami-Klein ball
        Input:
            v, a vector in ambient space coordinates, with nth dimension 'time-like'
        Output:
            a vector in Beltrami-Klein coordinates
    '''
    return (v/v[-1, :])

def project_from_klein(v):
    '''
        Project Beltrami-Klein ball points to hyperboloid
        Input:
            v, a vector, or array of vectors, in ambient space coordinates, with nth dimension = 1
        Output:
            a vector, or array of vectors, in hyperboloid coordinates
    '''
    if type(v) != np.ndarray:
        v = np.array(v)
    coeff = 1./np.sqrt(1-v[:-1,:]**2)

    return coeff*v
