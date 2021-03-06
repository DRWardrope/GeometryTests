import numpy as np
from geometry import *
def gradient_descent(
                        pt_i, target, differential_fn, 
                        geometry="hyperbolic", learning_rate=1.,
                        return_vectors=False,
                        test_rel_correction=False,
                    ):
    '''
        Calculate local gradient of differential, given the current pt and the target.
        Inputs:
                Two (d+1)-dimensional vectors in ambient space co-ordinates, pt_i and target
                pt_i: (d+1)-dimensional vector in ambient space co-ordinates,
                       the point to evaluate the gradient at.
                target: (d+1)-dimensional vectors in ambient space co-ordinates, the target point
                differential_fn: function that calculates the derivative
                learning_rate: dictates how far to step in gradient direction
    '''
#    print("gradient_descent({},{},{},{},{},{},{}):".format(
#                type(pt_i), type(target), differential_fn, geometry, learning_rate, return_vectors, test_rel_correction
#            )
#         )
    # Calculate gradient in ambient space co-ordinates
    step = differential_fn(pt_i, target, geometry)
    print("gradient_descent: step =",step)
    # Project this gradient onto tangent space
    projection = project_to_tangent(pt_i, step, geometry)
    print("gradient_descent: projection on tangent space = ",projection)
    # Map to manifold and return this updated pt
    if return_vectors:
        return (
                    exponential_map(-learning_rate*projection, pt_i, geometry),
                    step,
                    projection,
                )
    else:
        return exponential_map(-learning_rate*projection, pt_i, geometry)

def error_differential_eucl(u, v, geometry="hyperbolic"):
    '''
        Calculate differential of distance between points u and v, **with respect to u**,
        accounting for different geometries by implementing an appropriate metric.
        Inputs:
            u: (d+1)-dimensional vector, expressed in ambient space coordinates
            v: (d+1)-dimensional vector, expressed in ambient space coordinates
            geometry: specifies which metric to use (and hence how inner product calculated)
        Outputs:
            gradient of the distance in (d+1)-dimensional ambient space coordinates
    '''   
    if np.array_equal(u,v):
        return np.zeros(u.shape)
    # If u and v are different, calculate the gradient
    print("u = {}, v = {}, u.v = {}".format(u, v, dot(u, v, geometry)))
    if geometry == "spherical":
        coeff = -1./(np.sqrt(1.-dot(u, v, geometry)**2)+1e-10)
    elif geometry == "hyperbolic":        
        coeff = 1./(np.sqrt(dot(u, v, geometry)**2-1.)+1e10)
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))

    metric = get_metric(u.shape[0], geometry)
    return coeff*metric.dot(v)

def frechet_diff(p_eval, points, geometry="spherical"):
    '''
        Calculates the differential to enable a gradient descent algorithm to find 
        the Karcher/Fréchet mean of a set of points.
        Inputs:
            p_eval: Point at which to evaluate the derivative (usually a guess at 
                    the mean). (d+1)-dimensional vector, expressed in ambient space
                    coordinates.
            points: Array of n points which the derivative is calculate w.r.t. to. 
                    (d+1)*n-dimensional vector, expressed in ambient space
                    coordinates.
            geometry: string specifying which metric and distance function to use.
        Outputs:
            Derivative: (d+1)-dimensional vector, expressed in ambient space
                        coordinates.
    '''
    metric = get_metric(p_eval.shape[0], geometry)
#    update = np.zeros([p_eval.shape[0], 1])
    coeffs = -2.*distance(p_eval, points, geometry)
    print("numerator = ",coeffs)
    if geometry == "spherical":
        coeffs /= np.sqrt(1.-dot(p_eval, points, geometry)**2)+ 1.e-10
    elif geometry == "hyperbolic":
        coeffs /= np.sqrt(dot(p_eval, points, geometry)**2-1.)
    return np.atleast_2d(np.sum(coeffs*metric*points, axis=1)).T
    #return np.atleast_2d(np.sum(coeffs*points, axis=1)).T
