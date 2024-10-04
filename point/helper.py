
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from point.model import CoxLowRankSpatialModel
from point.low_rank.low_rank_deep import LowRankDeep

from point.misc import Space
from point.laplace import LaplaceApproximation

import gpflow.kernels as gfk
from gpflow.config import default_float

from enum import Enum


def defaultArgs():
    out = dict(length_scale = tf.constant([0.5], dtype=default_float(), name='lengthscale'),
               variance = tf.constant([5], dtype=default_float(), name='variance'),
               beta0 = None,
               kernel = "RBF"
               )
    
    return out

    
def get_lrgp(space = Space(), n_components = 250, n_layers = 2, m = [3.0, 5.0], d = [0.5, 1.0], sample = 200, n_dimension = 2, random_state = None, **kwargs):

    lrgp = None
    kwargs = {**defaultArgs(), **kwargs} #merge kwards with default args (with priority to args)
    length_scale = kwargs['length_scale']
    variance = kwargs['variance']
    beta0 = kwargs['beta0']
    kernel_str = kwargs['kernel']

    kernel = None
    kernel = gfk.SquaredExponential(variance= variance, lengthscales= length_scale)

    lrgp = LowRankDeep(kernel, beta0 = beta0, space = space, n_components =  n_components, n_dimension = n_dimension, n_layers = n_layers, sample = 200, m = m, d = d, random_state = random_state)
    lrgp.sample(latent_only = True)
    lrgp._is_fitted = True


    

    return lrgp



def get_process(name = None,  n_components = 250, n_layers = 2, m = [3.0, 5.0], d = [0.5, 1.0], sample = 200, n_dimension = 2, kernel = None, random_state = None, **kwargs):
    lrgp = get_lrgp( n_components = n_components, n_layers = n_layers, m = m, d = d, sample = sample, n_dimension = n_dimension, kernel = kernel, random_state = random_state, **kwargs)
    return CoxLowRankSpatialModel(lrgp, name, random_state = random_state)
        
    
def get_rff_model(name = "model", n_dims = 2, n_components = 75, n_layers = 2, m = [3.0, 5.0], d = [0.5, 1.0], sample = 200, variance = 2.0, space = Space(), kernel = None, random_state = None):
    
    name = name + ".rff." + str(n_components)
    variance = tf.Variable(variance, dtype=default_float(), name='sig')
    length_scale = tf.Variable(n_dims  * [0.5], dtype=default_float(), name='lenght_scale')

    model = get_process(
        name = name,
        length_scale = length_scale, 
        variance = variance, 
        space = space,
        n_components = n_components, 
        m = m,
        d = d,
        sample = sample,
        n_dimension = n_dims, 
        random_state = random_state,
        kernel = kernel,
        n_layers = n_layers
    )
    
    lp = LaplaceApproximation(model) 

    return lp



