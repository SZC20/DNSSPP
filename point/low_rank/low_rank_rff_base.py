
import tensorflow as tf
import numpy as np

import gpflow
from gpflow.config import default_float
from gpflow.base import Parameter
from gpflow.utilities import positive

from point.low_rank.low_rank_base import LowRankBase

from point.utils import check_random_state_instance
from point.misc import Space

from enum import Enum


class kernel_type(Enum):
    RBF = 1
    MATERN = 2
    
def kernel_type_cvrt(kernel):
    if type(kernel).__name__ == "SquaredExponential" :
        return (kernel_type.RBF, 0)
    elif type(kernel).__name__ == "Matern32" :
           return (kernel_type.MATERN, 3)
    elif type(kernel).__name__ == "Matern52" :
           return (kernel_type.MATERN, 5)
    elif type(kernel).__name__ == "Matern12" :
           return (kernel_type.MATERN, 1)
    else :
        return None



class LowRankRFFBase(LowRankBase):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = 250, n_dimension = 2,  random_state = None):

        # if n_dimension == 1 and not (kernel.lengthscales.shape == [] or kernel.lengthscales.shape[0] == 1):
        #     raise NotImplementedError("dimension of n_dimension:=" + str(n_dimension) + " not equal to legnscales_array_szie:=" + str(kernel.lengthscales.shape))
        # low_rank_deep dont satisfy this condition

        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
        (k_type, df) = kernel_type_cvrt(kernel)
        self.k_type = k_type
        self._df = df
        self._points_trainable = False

        # non Stationary part
        self.l11 = Parameter([1.0], transform=positive(), name = "l11")
        self.l12 = Parameter([1.0], transform=positive(), name = "l12")
        self.l22 = Parameter([1.0], transform=positive(), name = "l22")
        

        gpflow.set_trainable(self.l11, False)
        gpflow.set_trainable(self.l12, False)
        gpflow.set_trainable(self.l22, False)
        
        
    
    def __call__(self, X, X2 = None):
        if X2 is None :
            Z = self.feature(X)
            return Z @ tf.transpose(Z)
        return  self.feature(X)  @ tf.transpose(self.feature(X2))
 

        
    def set_points_trainable(self, trainable, nos = False):
        
        if not self._is_fitted :
            raise ValueError("object not fitted")
        
        if trainable is True :
            self._points_trainable = True
            if nos:
                self._G2 = tf.Variable(self._G2)
                # gpflow.set_trainable(self._G2, True)
            else:
                self._G = tf.Variable(self._G)
                # gpflow.set_trainable(self._G, True)
            # 
        else :
            self._points_trainable = False
            if nos:
                self._G2 = tf.constant(self._G2)
            else:
                self._G = tf.constant(self._G)
    
    
    

    def sample(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (self.n_features, 1)), dtype=default_float(), name='latent')
        if latent_only : return
        
        size = (self.n_dimension, self.n_components)
        self._G = tf.constant(random_state.normal(size = size), dtype=default_float(), name='G')

        if  self.k_type ==  kernel_type.MATERN :
            self._u = tf.constant(np.random.chisquare(self._df, size = (1, self.n_components)), dtype=default_float(), name='u')
        pass
    
    
    def fit(self, sample = True):
        if sample : self.sample()
        
        self.fit_random_weights()
        self._is_fitted = True
        return self


    def fit_random_weights(self):
        gamma = 1 / (2 * self.lengthscales **2 )

        if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
            self._random_weights =  tf.math.sqrt(2 * gamma) * self._G # 实际上应该乘以(2*pi*lengthscales)^-1，这里大概是不理参数的常数项了，反正最后会算出最优参数的。
        else :
            self._random_weights =  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ self._G
        # print("rw = ",self._random_weights.shape)
        
        if  self.k_type ==  kernel_type.MATERN :
            self._random_weights  *=  tf.math.sqrt(self._df / self._u) 
            
## NOS part

    def sample_nos(self, latent_only = False):
        random_state = check_random_state_instance(self._random_state)
        self._latent = tf.constant(random_state.normal(size = (self.n_features, 1)), dtype=default_float(), name='latent')
        if latent_only : return
        
        size2 = (self.n_dimension, 2, self.n_components) 
        self._G2 = tf.constant(random_state.normal(size = size2), dtype=default_float(), name='G2')
        # gpflow.set_trainable(self._G2, False)

        # if  self.k_type ==  kernel_type.MATERN :
        #     self._u = tf.constant(np.random.chisquare(self._df, size = (1, self.n_components)), dtype=default_float(), name='u')
        # pass
    
    
    def fit_nos(self, sample = True):

        gpflow.set_trainable(self.l11, True)
        gpflow.set_trainable(self.l12, True)
        gpflow.set_trainable(self.l22, True)
        # gpflow.set_trainable(self.lou, True)
        gpflow.set_trainable(self.lengthscales, False)
        
        if sample : self.sample_nos()
        
        self.fit_random_weights_nos()
        self._is_fitted = True
        return self


    def fit_random_weights_nos(self):

        l = tf.experimental.numpy.vstack([tf.experimental.numpy.hstack((self.l11, [0.0])), tf.experimental.numpy.hstack((self.l12, self.l22))])
        Sigma =  l @ tf.transpose(l)
        Sigmaminus = tf.linalg.inv(Sigma)
        # tmp = tf.linalg.triangular_solve(l, tf.eye(2, dtype = 'double') , lower=True)
        # Sigmaminus = tf.linalg.triangular_solve(tf.transpose(l), tmp , lower=True)
        L  = tf.linalg.cholesky(Sigmaminus)
        # print(L)
        self._random_weights =  tf.expand_dims(L,0) @ self._G2
        self.rw1 = tf.reshape(self._random_weights[:,0,:],(self.n_dimension,-1))
        self.rw2 = tf.reshape(self._random_weights[:,1,:],(self.n_dimension,-1))
        # print(self._random_weights)
        # print(self.rw1)
        # print(self.rw2)
        # gpflow.set_trainable(self.lengthscales, False)


        # gamma = 1 / (2 * self.lengthscales **2 )

        # if len(gamma.shape) == 0 or gamma.shape[0] == 1 :
        #     self._random_weights =  tf.math.sqrt(2 * gamma) * self._G 
        # else :
        #     # self._random_weights =  tf.linalg.diag(tf.math.sqrt(2 * gamma))  @ self._G
        #     raise ValueError("data shoule be 1d")
        
        # if  self.k_type ==  kernel_type.MATERN :
        #     self._random_weights  *=  tf.math.sqrt(self._df / self._u) 
 