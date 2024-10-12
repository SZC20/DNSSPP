import numpy as np
from sklearn.model_selection import train_test_split
from point.laplace import opt_method

from point.helper import get_rff_model
from point.optim.optim_function import  get_optim_func_deep
from point.misc import Space
from point.utils import build_coal_data, domain_grid_1D, domain_grid_2D


## synthetic data 1
test_X = np.linspace(0, 10, 1000).reshape(-1, 1)
event_time = np.load('events.npy')
intensity = np.load('intensity.npy').reshape(-1,1)

rng = np.random.RandomState(120)

scale = 1
domain = [0,10]

X = np.load('intensity_X.npy',allow_pickle=True).tolist()
space =  Space(scale = scale) 
grid =  domain_grid_1D(space.bound1D, 1000)
# ogrid =  domain_grid_1D(domain, 200)
shift = (domain[1]-domain[0])/space.measure(1)

ltest_mean = {}
l2_mean = {}
t_mean = {}
ltest_std = {}
l2_std = {}

## DSSPP-1

md_ltest = []
md_l2 = []
md_t = []

for i in range(10):
    X_train = X[i]
    X_test = X[:i] + X[i+1:]

    md = get_rff_model(name = 'deep', n_components = [100, 50], n_layers=2, sample = 1000, m = [0,0], d = [1, 1], n_dims = 1, space = space, random_state = rng)
    md.lrgp.beta0.assign( md.lrgp.beta0 ) 
    ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
    opt_time = ofunc(md, X_train, verbose = True)
    md_t.append(opt_time)

    ld = md.predict_lambda(grid) / shift
    l2 = sum((intensity-ld)**2).numpy()
    l2 = np.sqrt(l2[0]) / 1000
    md_l2.append(l2)

    for x in X_test:
        ldp = md.predictive_log_likelihood(x)
        md_ltest.append(ldp.numpy())

    
ltest_mean['md'] = np.array(md_ltest).mean()
l2_mean['md'] = np.array(md_l2).mean()
t_mean['md'] = np.array(md_t).mean()
ltest_std['md'] = np.std(md_ltest, ddof=1)
l2_std['md'] = np.std(md_l2, ddof=1)

    

## synthetic data 2
test_X = np.linspace(0, 10, 1000).reshape(-1, 1)
event_time = np.load('events_nos.npy')
intensity = np.load('intensity_nos.npy').reshape(-1,1)

rng = np.random.RandomState(120)

scale = 1
domain = [0,10]

X = np.load('intensity_nos_X.npy',allow_pickle=True).tolist()
space =  Space(scale = scale) 
grid =  domain_grid_1D(space.bound1D, 1000)
# ogrid =  domain_grid_1D(domain, 200)
shift = (domain[1]-domain[0])/space.measure(1)

ltest_mean = {}
l2_mean = {}
t_mean = {}
ltest_std = {}
l2_std = {}

md_ltest = []
md_l2 = []
md_t = []

for i in range(10):
    X_train = X[i]
    X_test = X[:i] + X[i+1:]

    md = get_rff_model(name = 'deep', n_components = [100, 50], n_layers=2, sample = 1000, m = [0,0], d = [0.8, 0.8], n_dims = 1, space = space, random_state = rng)
    md.lrgp.beta0.assign( md.lrgp.beta0 ) 
    ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
    opt_time = ofunc(md, X_train, verbose = True)
    md_t.append(opt_time)

    ld = md.predict_lambda(grid) / shift
    l2 = sum((intensity-ld)**2).numpy()
    l2 = np.sqrt(l2[0]) / 1000
    md_l2.append(l2)

    for x in X_test:
        ldp = md.predictive_log_likelihood(x)
        md_ltest.append(ldp.numpy())

    
ltest_mean['md'] = np.array(md_ltest).mean()
l2_mean['md'] = np.array(md_l2).mean()
t_mean['md'] = np.array(md_t).mean()
ltest_std['md'] = np.std(md_ltest, ddof=1)
l2_std['md'] = np.std(md_l2, ddof=1)

## Coal
def normalize(X, domain, scale = 1.0) :
    center = np.mean(domain)
    norm = domain[1] - center
    return scale * (X - center) / norm

rng = np.random.RandomState(150)

scale = 2
X, domain = build_coal_data()
X =  normalize(X, domain, scale)
space =  Space(scale = scale) 
X_train, X_test = train_test_split(X, test_size= 0.5,  random_state = rng)

ltest_coal = {}
t_coal = {}
ltest_coal_std = {}

md_ltest = []
md_t = []

for i in range(10):
    X_train, X_test = train_test_split(X, test_size= 0.5,  random_state = rng)

    md = get_rff_model(name = 'deep', n_components = [100, 50], sample = 1000, m = [0,0], d = [0.3,0.3], n_dims = 1,  space = space, random_state = rng)
    md.lrgp.beta0.assign( md.lrgp.beta0 ) 
    md.lrgp.variance.assign(0.1)
    ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
    
    opt_time = ofunc(md, X_train, verbose = True)
    llp = md.predictive_log_likelihood(X_test)

    md_ltest.append(llp.numpy())
    md_t.append(opt_time)

ltest_coal['md'] = np.array(md_ltest).mean()
t_coal['md'] = np.array(md_t).mean()
ltest_coal_std['md'] = np.std(md_ltest, ddof=1)


## Redwoods
def get_rw_data_set(scale = 1.0):
    name = 'redwood'
    directory = "./data"
    data = np.genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = scale * data[1:, 0:2]
    return data

rng  = np.random.RandomState(125)
X = get_rw_data_set(scale = 2)
space = Space(scale = 2)

variance = 0.1
l = [0.3, 0.3]
tol = 1e-06

ltest_rw = {}
ltest_rw_std = {}
t_rw = {}

md_ltest = []
md_t = []

for i in range(10):
    X_train, X_test = train_test_split(X, test_size= 0.5,  random_state = rng)

    md = get_rff_model(name = 'deep', n_dims = 2, n_components = [100, 50], sample = 1000, m = [0,0], d = [0.00001, 0.00001], space = space, random_state = rng)
    md.lrgp.beta0.assign( md.lrgp.beta0 ) 
    md.lrgp.variance.assign(variance) 

    md.default_opt_method = opt_method.NEWTON_CG
    ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 30, wage_decay = True)
    
    opt_time = ofunc(md, X_train, verbose = True)
    llp = md.predictive_log_likelihood(X_test)

    md_ltest.append(llp.numpy())
    md_t.append(opt_time)

ltest_rw['md'] = np.array(md_ltest).mean()
t_rw['md'] = np.array(md_t).mean()
ltest_rw_std['md'] = np.std(md_ltest, ddof=1)

## Taxi
def normalize(X, domain, scale = 1.0) :
    center = np.mean(domain)
    norm = domain[1] - center
    return scale * (X - center) / norm

def get_taxi_data_set(scale = 1.0):
    rng  = np.random.RandomState(200)
    directory = "./data"
    name = "porto_trajectories"
    data = np.genfromtxt(directory + "/" + name + ".csv", delimiter=',')
    data = data[data[:,1] <= 41.18]
    data = data[data[:,1] >= 41.147]
    data = data[data[:,0] <= -8.58]
    data = data[data[:,0] >= -8.65]

    data = data[rng.choice(data.shape[0], 3000, replace=False), :]

    X1 =  normalize(data[:,0],  [-8.58, -8.65], scale = 1.0)
    X2 =  normalize(data[:,1],  [41.18, 41.147], scale = 1.0)
    data = scale * np.column_stack((X1,X2))
    
    return data

rng  = np.random.RandomState(200)
scale = 2
X = get_taxi_data_set(scale = scale)
space = Space(scale = scale)
shift = 15

models = []
variance = 0.1
l = [0.3, 0.3]
tol = 1e-06

ltest_taxi = {}
t_taxi = {}
ltest_taxi_std = {}

md_ltest = []
md_t = []

for i in range(10):
    X_train, X_test = train_test_split(X, test_size= 0.5,  random_state = rng)

    md = get_rff_model(name = 'deep', n_dims = 2, n_components = [100, 50], sample = 1000, m = [0,0], d = [1.5, 1.5], space = space, random_state = rng)
    md.lrgp.beta0.assign( md.lrgp.beta0 ) 
    md.lrgp.variance.assign( 0.05 ) 
    md.default_opt_method = opt_method.NEWTON_CG
    ofunc = get_optim_func_deep(n_loop = 1, maxiter = 25, xtol = 1e-05, epoch = 50, wage_decay = True)
    
    opt_time = ofunc(md, X_train, verbose = True)
    llp = md.predictive_log_likelihood(X_test)

    md_ltest.append(llp.numpy())
    md_t.append(opt_time)

ltest_taxi['md'] = np.array(md_ltest).mean()
t_taxi['md'] = np.array(md_t).mean()
ltest_taxi_std['md'] = np.std(md_ltest, ddof=1)


