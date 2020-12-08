import numpy as np
import jax
import symjax
import symjax.tensor as T




def RNTK_first(x,sw,su,sb,sh,L,Lf,sv):
    X = x*x[:, None]
    n = X.shape[0] #
    gp_new = T.expand_dims(sh ** 2 * sw ** 2 * T.eye(n, n) + (su ** 2) * X + sb ** 2, axis = 0)
    rntk_new = gp_new
    for l in range(L-1):
        l = l+1
        S_new,D_new = VT(gp_new[l-1])
        gp_new = T.concatenate([gp_new,T.expand_dims(sh ** 2 * sw ** 2 * T.eye(n, n) + su**2 * S_new + sb**2,axis = 0)])
        rntk_new = T.concatenate([rntk_new,T.expand_dims(gp_new[l] + (Lf <= (l-1))*su**2*rntk_new[l-1]*D_new,axis = 0)])
    S_old,D_old = VT(gp_new[L-1])
    gp_new = T.concatenate([gp_new,T.expand_dims(sv**2*S_old,axis = 0)])
    rntk_new = T.concatenate([rntk_new,T.expand_dims(gp_new[L] + (Lf != L)*sv**2*rntk_new[L-1]*D_old,axis = 0)])
    return rntk_new, gp_new
def RNTK_middle(previous,x,sw,su,sb,L,Lf,sv):
    X = x * x[:, None]
    rntk_old = previous[0]
    gp_old = previous[1]
    S_old,D_old = VT(gp_old[0])
    gp_new = T.expand_dims(sw ** 2 * S_old + (su ** 2) * X + sb ** 2,axis = 0)
    if Lf == 0:
        rntk_new = T.expand_dims(gp_new[0] + sw**2*rntk_old[0]*D_old,axis = 0)
    else:
        rntk_new = T.expand_dims(gp_new[0],axis = 0)
    for l in range(L-1):
        l = l+1
        S_new,D_new = VT(gp_new[l-1])
        S_old,D_old = VT(gp_old[l])
        gp_new = T.concatenate( [gp_new, T.expand_dims( sw ** 2 * S_old + su ** 2 * S_new +  sb ** 2, axis = 0)])
        rntk_new = T.concatenate( [ rntk_new,  T.expand_dims( gp_new[l] +(Lf <= l)*sw**2*rntk_old[l]*D_old +(Lf <= (l-1))* su**2*rntk_new[l-1]*D_new  ,axis = 0)   ]  )
    S_old,D_old = VT(gp_new[L-1])
    gp_new = T.concatenate([gp_new,T.expand_dims(sv**2*S_old,axis = 0)])
    rntk_new = T.concatenate([rntk_new,T.expand_dims(rntk_old[L]+ gp_new[L] + (Lf != L)*sv**2*rntk_new[L-1]*D_old,axis = 0)])
    return T.stack([rntk_new,gp_new]),x

def RNTK_output(previous,sv,L,Lf,length):
    rntk_old = previous[0]
    gp_old = previous[1]
    S_old,D_old = VT(gp_old[L-1])
    RNTK_last  = sv**2*S_old + (Lf != L)*sv**2*rntk_old[L-1]*D_old 
    RNTK_avg =  (RNTK_last + rntk_old[L])/length
    return RNTK_last,RNTK_avg

def RNTK_function(N,length,param):
    DATA = T.Placeholder((N, length), 'float32')
    RNTK,GP = RNTK_first(DATA[:,0], param['sigmaw'],param['sigmau'],param['sigmab'],param['sigmah'],param['L'], param['Lf'],param['sigmav'])
    v, _ = T.scan(lambda a,b:RNTK_middle(a,b,param['sigmaw'],param['sigmau'],param['sigmab'],param['L'], param['Lf'],param['sigmav']
                                         ),sequences=[ T.transpose(DATA[:, 1:]) ], init=T.stack([RNTK,GP]))
    RNTK_last,RNTK_avg = RNTK_output(v, param['sigmav'],param['L'],param['Lf'],length)
    f = symjax.function(DATA, outputs= [RNTK_last,RNTK_avg])
    return f


def VT(M):
    A = T.diag(M)  # GP_old is in R^{n*n} having the output gp kernel
    # of all pairs of data in the data set
    B = A * A[:, None]
    C = T.sqrt(B)  # in R^{n*n}
    D = M / C  # this is lamblda in ReLU analyrucal formula
    E = T.clip(D, -1, 1)  # clipping E between -1 and 1 for numerical stability.
    F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
    G = (np.pi - T.arccos(E)) / (2 * np.pi)
    return F,G