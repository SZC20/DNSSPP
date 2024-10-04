
import time
import numpy as np
from point.laplace import opt_type
import copy

def get_optim_func_deep(n_loop = 1, maxiter = None, xtol = None, lmin = 1e-05, lmax = 10.0, epoch = 100, smax = None, smin = None, num_attempt = 5, direct_grad = False, nos=False, wage_decay = False, learning_rate = 0.001) :

    _opt = opt_type.AUTO_DIFF
    if direct_grad is True : _opt = opt_type.DIRECT

    def optim_func(model, X, verbose = False):

        beta = np.sqrt(X.shape[0] / model.lrgp.space_measure)
        model.lrgp.set_drift(beta, trainable = True) # 注意，为什么在这里设置beta？
        model.set_X(X)
        
        opt_t = 0
        opt_mll = -1000
        opt_model = None

        # init_params = model.lrgp.copy_params()
        
        if isinstance(maxiter, list):
            m_p = maxiter[0]
            m_m = maxiter[1]
        else :
            m_p = m_m = maxiter

        for r in range(1):
    
            # ℓmin_active = True
            # ℓmax_active = True
            s_active = True
            n_iter = 0

            while s_active and n_iter < num_attempt :
                
                t0 = time.time()
    
                if verbose and n_iter > 0 : 
                    print("optim_restart_" + str(n_iter)) 
                    
                # model.lrgp.reset_params(init_params, sample = True)
                
                # try :

                model.optimize_mode(optimizer = opt_type.DIRECT, tol = xtol, verbose = verbose)
                # model.optimize_mode(optimizer = opt_type.AUTO_DIFF, tol = xtol, verbose = verbose)
                # print(111111111)

                for i in range(n_loop):
                    model.optimize_deep(optimizer = _opt, m = m_p, verbose = verbose, epoch = epoch, nos = nos, wage_decay = wage_decay, learning_rate = learning_rate)
                    # print(22222222222)
                    model.optimize_mode(optimizer = opt_type.DIRECT, m = m_m, tol = xtol, verbose = verbose) 
                    # model.optimize_mode(optimizer = opt_type.AUTO_DIFF, m = m_m, tol = xtol, verbose = verbose) 
                    # print(33333333)


                t = time.time() - t0
                # ℓmin_active = np.any(model.lrgp.lengthscales.numpy() < lmin )
                # ℓmax_active = np.any(model.lrgp.lengthscales.numpy() > lmax )
                
                s_active = False
                if smax is not None :
                    s_active = (model.smoothness_test().numpy() > smax)
                if smin is not None :
                    s_active =  (model.smoothness_test().numpy() < smin)

                mll = model.log_marginal_likelihood()
                if verbose : print("mll:= %f" % (mll[0].numpy()))
                n_iter +=1
    
                # if mll > opt_mll and not s_active  :
                opt_mll = mll
                # opt_model = copy.deepcopy(model)
                opt_t += t
                    
                # except BaseException as err:
                #             msg_id = "model#"  + model.name + ": Optim ERROR stopped and re-attempt"
                #             msg_e = f"Unexpected {err=}, {type(err)=}"
                #             if verbose : 
                #                 print(msg_id) 
                #                 print(msg_e )

        # model.copy_obj(opt_model)
        t = opt_t

        if verbose :
            print("SLBPP(" + model.name + ") finished_in := [%f] " % (t))
  
        return t

    return optim_func
