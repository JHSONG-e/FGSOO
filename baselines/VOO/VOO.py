import numpy as np
from scipy.stats import norm
from scipy.special import erf 
from scipy.spatial import distance


def sqrt_safe(x,eps=1e-6):
    return np.sqrt(np.abs(x)+eps)

def x_sampler(n_sample,x_minmax):
    """
    Sample x as a list from the input domain 
    """
    x_samples = []
    for _ in range(n_sample):
        x_sample = x_minmax[:,0]+(x_minmax[:,1]-x_minmax[:,0])*np.random.rand(1,x_minmax.shape[0])
        x_samples.append(x_sample)
    return x_samples # list 


def r_sq(x1,x2,x_range=1.0,invlen=5.0):
    """
    Scaled pairwise dists 
    """
    x1_scaled,x2_scaled = invlen*x1/x_range,invlen*x2/x_range
    D_sq = distance.cdist(x1_scaled,x2_scaled,'sqeuclidean') 
    return D_sq


def get_best_xy(x_data,y_data):
    """
    Get the current best solution
    """
    min_idx = np.argmax(y_data)
    return x_data[min_idx,:].reshape((1,-1)),y_data[min_idx,:].reshape((1,-1))

def sample_from_best_voronoi_cell(x_data,y_data,x_minmax,
                                  max_try_sbv=5000):
    """
    Sample from the Best Voronoi Cell for Voronoi Optimistic Optimization (VOO)
    """
    x_dim = x_minmax.shape[0]
    idx_min_voo = np.argmax(y_data) # index of the best x

    n_try,x_tried,d_tried = 0,np.zeros((max_try_sbv,x_dim)),np.zeros((max_try_sbv,1))
    x_sol,_ = get_best_xy(x_data,y_data)
    x_evals = []
    rand = np.random.rand()
    while True:
        # if n_try < (max_try_sbv/2):
        x_sel = x_sampler(n_sample=1,x_minmax=x_minmax)[0] # random sample
        if rand < 0.5:
            break 
        else:
            # Gaussian sampling centered at x_sel
            dist_sel = r_sq(x_data,x_sel)
            idx_min_sel = np.argmin(dist_sel)
            if idx_min_sel == idx_min_voo: 
                break
            # Sampling the best vcell might took a lot of time 
            x_tried[n_try,:] = x_sel
            d_tried[n_try,:] = r_sq(x_data[idx_min_voo,:].reshape((1,-1)),x_sel)
            n_try += 1 # increase tick
            if n_try >= max_try_sbv:
                idx_min_tried = np.argmin(d_tried) # find the closest one 
                x_sel = x_tried[idx_min_tried,:].reshape((1,-1))
                break
    x_evals.append(x_sel) # append 

    return x_evals