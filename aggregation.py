import numpy as np
from geometry import project_from_klein, project_to_klein

def einstein_midpoint(points):
    klein_pts = project_to_klein(points)
    gammas = 1./np.sqrt(1-np.sum(klein_pts[:-1,:]**2, axis=0))
    klein_ein_midpt = np.atleast_2d(
                        np.sum(gammas*klein_pts, axis=1)
                     ).T/np.sum(gammas)
    return project_from_klein(klein_ein_midpt)
