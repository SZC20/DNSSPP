
import numpy as np
import copy
import math

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from gpflow.config import default_float
import gpflow.kernels as gfk 

from point.low_rank.low_rank_rff_base import LowRankRFFBase
from point.misc import Space
from point.utils import domain_grid_2D



# def expandedSum(x):
#     z1 = tf.expand_dims(x, 1)
#     # print(z1.shape)
#     z2 = tf.expand_dims(x, 0)
#     # print(z2.shape)
#     return (z1 + z2, z1 - z2)

# def expandedSum_cross(x1,x2):
#     z1 = tf.expand_dims(x1, 1)
#     z2 = tf.expand_dims(x2, 0)
#     Mp = z1 + z2
#     Mm = z1 - z2
#     d = tf.stack([Mp[:,:,0] , Mm[:,:,0] ])
#     return d

# def expandedSum2D(x):
#     Mp, Mm = expandedSum(x)
#     d1 = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(x.shape[0],  dtype=default_float())) ])
#     d2 = tf.stack([Mp[:,:,1] , tf.linalg.set_diag(Mm[:,:,1], tf.ones(x.shape[0],  dtype=default_float())) ])
#     return (d1, d2)

# def expandedSum1D(x):
#     Mp, Mm = expandedSum(x)
#     # print(Mp.shape,Mm.shape)
#     d = tf.stack([Mp[:,:,0] , tf.linalg.set_diag(Mm[:,:,0], tf.ones(x.shape[0],  dtype=default_float())) ])
#     return d

# 先不考虑维度为2的情况

class DSKN_n(tf.keras.Model):

    def __init__(self, n_components, n_dims = 1, n_layers = 2, m = [3.0, 5.0], d = [0.5, 1.0], variance = 1.0):

        super(DSKN_n, self).__init__(name='DSKN_n')
        self.n_components = n_components
        self.n_layers = n_layers
        self.n_dims = n_dims
        self.variance = variance
        if self.n_layers != len(self.n_components):
            raise Exception("The length of n_components is not equal to n_layers.")
        
        self.op_list0 = [tf.keras.layers.Dense(self.n_components[0], input_shape = (self.n_dims,),  kernel_initializer= tf.keras.initializers.RandomNormal(
  mean=m[0], stddev=d[0]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None))]
        self.op_list1 = [tf.keras.layers.Dense(self.n_components[0], input_shape = (self.n_dims,), kernel_initializer= tf.keras.initializers.RandomNormal(
  mean=m[0], stddev=d[0]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None))]

        for i in range(self.n_layers - 1):
            self.op_list0.append(tf.keras.layers.Dense(self.n_components[i+1], kernel_initializer= tf.keras.initializers.RandomNormal(
  mean=m[i+1], stddev=d[i+1]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None)))
            self.op_list1.append(tf.keras.layers.Dense(self.n_components[i+1], kernel_initializer= tf.keras.initializers.RandomNormal(
  mean=m[i+1], stddev=d[i+1]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None)))

#         self.op_list0 = [tf.keras.layers.Dense(self.n_components[0], kernel_initializer= tf.keras.initializers.RandomNormal(
#   mean=m[0], stddev=d[0]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None)), tf.keras.layers.Dense(self.n_components[1],  kernel_initializer= tf.keras.initializers.RandomNormal(
#   mean=m[1], stddev=d[1]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None))]
#         self.op_list1 = [tf.keras.layers.Dense(self.n_components[0],  kernel_initializer= tf.keras.initializers.RandomNormal(
#   mean=m[0], stddev=d[0]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None)), tf.keras.layers.Dense(self.n_components[1],  kernel_initializer= tf.keras.initializers.RandomNormal(
#   mean=m[1], stddev=d[1]), bias_initializer= tf.keras.initializers.RandomUniform(minval=0 , maxval= 2*math.pi, seed=None))]


    
    def call(self, x):
        # print(x)
        # x = tf.keras.layers.Flatten()(x) 
        if self.n_dims == 2:
            x = tf.reshape(x, [-1,2])
        else:
            x = tf.reshape(x, [-1,1])

        for i in range(self.n_layers):
            x = tf.cos(self.op_list0[i](x)) + tf.cos(self.op_list1[i](x))
        feature_mapping =  np.sqrt(self.variance /(2 * self.n_components[-1])) * x
        
        return feature_mapping

class LowRankDeep(LowRankRFFBase):
    
    def __init__(self, kernel, beta0 = None, space = Space(), n_components = [10,10], n_layers = 2, n_dimension = 1, m = [3.0, 5.0], d = [0.5, 1.0], sample = 200,  random_state = None):

        super().__init__(kernel, beta0, space, n_components, n_dimension, random_state)
            
        self._points_trainable = True
        self.inte_sample = sample
        self.network = DSKN_n(n_components = self.n_components, m = m, d = d, n_dims = self.n_dimension, n_layers = n_layers, variance = self.variance)
        
    @property
    def n_features(self):
        return self.n_components[-1]
        
    
    def copy_params(self):
        tpl = list((self.network.trainable_variables, self.beta0, self._points_trainable))
        return copy.deepcopy(tpl)

    def reset_params(self, p, sample = True):
        self.network.trainable_variables.assign(p[0])
        self.beta0.assign(p[1])
        self.set_points_trainable(p[2])
        # self.fit_nos(sample = sample)

    
    def feature(self, X):
        """ Transforms the data X (n_samples, n_dimension) 
        to feature map space Z(X) (n_samples, n_components)"""
        
        feature =  tf.cast(self.network(X), tf.float64)
        return feature
    


    def M(self):
        if self.n_dimension == 2:
            return self.__M_2D()
        else:
            return self.__M_1D()
    
    
    
    def m(self):

        if self.n_dimension == 2 :
            m = self.__m_2D()
        else :
            m = self.__m_1D()
  
        return m



    def integral(self, bound = None, get_grad = False, full_output = False):
        
        if bound is None :
            bound = self.space.bound1D
            
        if self.n_dimension == 2 :
            M  = self.__M_2D()
        else:
            M  = self.__M_1D()

        # M = tf.dtypes.cast(M, tf.float64)
        integral = tf.transpose(self._latent) @ M @ self._latent
        
        add_to_out = 0.0
        sub_to_out = 0.0

        if self.hasDrift is True :
            
            if self.n_dimension == 2 :
                m = self.__m_2D()
            else :
                m = self.__m_1D()

            # m = tf.dtypes.cast(m, tf.float64)            
            # m = tf.reshape(m, [-1,1])
            beta_term = 2 * self.beta0 * tf.transpose(self._latent) @ m
            integral += beta_term 
            add_to_out = self.beta0**2 *self.space_measure
            sub_to_out = beta_term[0][0]
 
        return integral[0][0] + add_to_out

    
    # def __M_2D(self, bound, get_grad = False):
    #     # Return the matrices B and A
    #     # without grad return : M = [B,A] (i.e. 2xRxR tensor)
    #     # with grad return : M = [[B, der1B, der2B], [A, der1A, der2A]] (i.e. 2x3xRxR tensor)

    #     if not self._is_fitted :
    #         raise ValueError("instance not fitted")
            
    #     if bound[0] != -bound[1]  :
    #         raise ValueError("implmentation only for symetric bounds")
            
    #     bound = bound[1]
    #     R =  self.n_components 
    #     z1 = self._random_weights[0,:]
    #     z2 = self._random_weights[1,:]

    #     d1, d2 =  expandedSum2D(tf.transpose(self._random_weights))

    #     sin_d1 = tf.sin(bound*d1) 
    #     sin_d2 = tf.sin(bound*d2)

    #     M = (2 / (d1 * d2)) * sin_d1  * sin_d2
    #     diag = tf.stack([(1 / (2 * z1* z2)) * tf.sin(2*bound*z1) * tf.sin(2 *bound*z2), 2 * bound ** 2 * tf.ones(R,  dtype=default_float())])                                     
    #     M = tf.linalg.set_diag(M, diag) 

    #     if get_grad :

    #         if self.lengthscales.shape == [] or self.lengthscales.shape == 1  :
    #             l1 = self.lengthscales
    #             l2 = self.lengthscales
    #         else :
    #             l1 = self.lengthscales[0]
    #             l2 = self.lengthscales[1]

    #         dl1 = - ( 2 * bound * tf.cos(bound*d1) * sin_d2 / d2 - M ) / l1
    #         dl1 =  tf.linalg.set_diag(dl1, tf.stack([ - ( bound * tf.cos(2*bound*z1) * tf.sin(2*bound*z2) / z2 - diag[0,:] ) / l1, tf.zeros(R,  dtype=default_float())]) ) 

    #         dl2 = - ( 2 * bound * sin_d1 * tf.cos(bound*d2) / d1 - M ) / l2
    #         dl2 =  tf.linalg.set_diag(dl2, tf.stack([ - ( bound * tf.sin(2*bound*z1) * tf.cos(2*bound*z2) / z1 - diag[0,:] ) / l2, tf.zeros(R,  dtype=default_float())])) 

    #         out = tf.experimental.numpy.vstack([
    #             tf.expand_dims( tf.experimental.numpy.vstack([tf.expand_dims(M[0,:],0), tf.expand_dims(dl1[0,:],0), tf.expand_dims(dl2[0,:],0)]),0),
    #             tf.expand_dims(tf.experimental.numpy.vstack([tf.expand_dims(M[1,:],0), tf.expand_dims(dl1[1,:],0), tf.expand_dims(dl2[1,:],0)]),0)
    #             ])

    #         return self.variance * out / R
            
    #     return self.variance * M / R
    
    
    def __M_1D(self):

        xi = tf.linspace(self.space.bound1D[0], self.space.bound1D[1], num = self.inte_sample)
        Mi = self.network(xi)
        M = tf.transpose(Mi) @ Mi
        M = M * (self.space.bound1D[1] - self.space.bound1D[0]) / self.inte_sample
        M = tf.cast(M, tf.float64)
        return M
    
    def __M_2D(self):
        step = (self.space.bound1D[1] - self.space.bound1D[0]) / self.inte_sample
        grid, _,_ = domain_grid_2D(bound = self.space.bound1D , step = step)
        Mi = self.network(grid)
        M = tf.transpose(Mi) @ Mi
        M = M * step * step
        M = tf.cast(M, tf.float64)
        return M

 
    
    
    # def __m_2D(self, bound, get_grad = False):
    #     # Return the vector m
    #     # without grad return : m (i.e. (0.5*R)x1 tensor)
    #     # with grad return : M = [m, der1m, der2m] (i.e. 3x(0.5*R)x1 tensor)

    #     if not self._is_fitted :
    #         raise ValueError("instance not fitted")
            
    #     if bound[0] != -bound[1]  :
    #         raise ValueError("implmentation only for symetric bounds")
            
    #     bound = bound[1]

    #     R =  self.n_components
    #     z1 = self._random_weights[0,:]
    #     z2 = self._random_weights[1,:]
        
    #     sin_z1 = tf.sin(bound*z1) 
    #     sin_z2 = tf.sin(bound*z2) 

    #     vec = 4* sin_z1 * sin_z2 
    #     vec =  tf.linalg.diag(1 / (z1 * z2)) @ tf.expand_dims(vec, 1)
    #     factor = tf.sqrt(tf.convert_to_tensor( self.variance/ R, dtype=default_float()))

    #     if get_grad is True :
            
    #         if self.lengthscales.shape == [] :
    #             l1 = l2 = self.lengthscales
    #         else :
    #             l1 = self.lengthscales[0]
    #             l2 = self.lengthscales[1]
                
    #         dl1 =  - ( np.expand_dims(4 * bound * tf.cos(bound*z1) * sin_z2 / z2,1) - vec ) / l1
    #         dl2 =  - ( np.expand_dims(4 * bound * sin_z1 * tf.cos(bound*z2) / z1,1) - vec ) / l2
    #         return factor * tf.experimental.numpy.vstack([tf.expand_dims(vec,0), tf.expand_dims(dl1,0), tf.expand_dims(dl2,0)])

    #     return  factor * vec
    
    
    
    def __m_1D(self):
        
        xi = tf.linspace(self.space.bound1D[0], self.space.bound1D[1], num = self.inte_sample)
        mi = self.network(xi)
        m = tf.reduce_sum(mi,0)
        m = m * (self.space.bound1D[1] - self.space.bound1D[0]) / self.inte_sample
        m = tf.reshape(m, [-1,1])
        m = tf.cast(m, tf.float64)
        return m
    
    def __m_2D(self):
        step = (self.space.bound1D[1] - self.space.bound1D[0]) / 1000
        grid, _,_ = domain_grid_2D(bound = self.space.bound1D , step = step)
        mi = self.network(grid)
        m = tf.reduce_sum(mi,0)
        m = m * step * step
        m = tf.reshape(m, [-1,1])
        m = tf.cast(m, tf.float64)
        return m






