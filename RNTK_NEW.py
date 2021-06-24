import numpy as np
import jax
import symjax
import symjax.tensor as T

class RNTK():
    def __init__(self, dic):
        self.sw = 1
        self.su = 1
        self.sb = 1
        self.sh = 1
        self.L = 1
        self.Lf = 0
        self.sv = 1
        self.N = int(dic["n_patrons1="])
        self.length = int(dic["n_entradas="])
        
    def VT(self, M):
        A = T.diag(M)  # GP_old is in R^{n*n} having the output gp kernel
        # of all pairs of data in the data set
        B = A * A[:, None]
        C = T.sqrt(B)  # in R^{n*n}
        D = M / C  # this is lamblda in ReLU analyrucal formula
        E = T.clip(D, -1, 1)  # clipping E between -1 and 1 for numerical stability.
        F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
        G = (np.pi - T.arccos(E)) / (2 * np.pi)
        return F,G

def make_inputs(dim_1, dim_2, dim1idx, dim2idx, n):
    diagindex = jax.numpy.arange(0,min(dim_1, dim_2) - (dim1idx + dim2idx))
    diag = T.Variable((diagindex), "float32", "dimension internal index")
    dim1ph = T.Variable(dim_1, "float32", "dimension 1 max")
    dim2ph = T.Variable(dim_2, "float32", "dimension 2 max")
    dim1idxph = T.Variable(dim1idx, "float32", "dimension 1 index")
    dim2idxph = T.Variable(dim2idx, "float32", "dimension 2 index")
    nph = T.Variable(n, "float32", "n")
    return diag, dim1ph, dim2ph, dim1idxph, dim2idxph, nph

def create_func_for_diag(rntk, dim_1, dim_2, dim1idx, dim2idx, n, function = False):
    diag, dim1ph, dim2ph, dim1idxph, dim2idxph, nph = make_inputs(dim_1, dim_2, dim1idx, dim2idx, n)

    ## prev_vals - (2,1) - previous phi and lambda values
    ## idx - where we are on the diagonal
    ## d1idx - y value of first dimension diag start
    ## d2idx - x value of second dimension diag start
    ## d1ph - max value of first dimension
    ## d2ph - max value of second dimension
    bc = rntk.sh ** 2 * rntk.sw ** 2 * T.eye(n, n) + (rntk.su ** 2) + rntk.sb ** 2 ## took out an X
    single_boundary_condition = T.expand_dims(T.Variable((bc), "float32", "boundary_condition"), axis = 0)
    boundary_condition = T.concatenate([single_boundary_condition, single_boundary_condition])

    def fn(prev_vals, idx, d1ph, d2ph, d1idx, d2idx, nph):
        # tiprime_iter = d1idx + idx
        # ti_iter = d2idx + idx
        prev_lambda = prev_vals[0]
        prev_phi = prev_vals[1]
        ## not boundary condition
        S, D = rntk.VT(prev_lambda)
        new_lambda = rntk.sw ** 2 * S + rntk.su ** 2 + rntk.sb ** 2 ## took out an X
        new_phi = new_lambda + rntk.sw ** 2 * prev_phi * D
        lambda_expanded = T.expand_dims(new_lambda, axis = 0)
        phi_expanded = T.expand_dims(new_phi, axis = 0)
        to_return = T.concatenate([lambda_expanded, phi_expanded])
        
        return to_return, to_return

    last_ema, all_ema = T.scan(
        fn, init = boundary_condition, sequences=[diag], non_sequences=[dim1ph, dim2ph, dim1idxph, dim2idxph,  nph]
    )

    expanded_ema = T.concatenate([T.expand_dims(boundary_condition, axis = 0), all_ema])
    if function: 
        f = symjax.function(diag, dim1ph, dim2ph, dim1idxph, dim2idxph, nph, outputs=expanded_ema)
        return f
    else:
        return expanded_ema

def diag_func_wrapper(rntk, dim_1, dim_2, dim_1_idx, dim_2_idx, n, fbool = False):
    f = create_func_for_diag(rntk, dim_1, dim_2, dim_1_idx, dim_2_idx, n, function = fbool)
    if fbool:
        return f(np.arange(0,min(dim_1, dim_2) - (dim_1_idx + dim_2_idx)), dim_1, dim_2, dim_1_idx, dim_2_idx, n)
    return f

def index_func(whic, dim_1, dim_2):
    dim = min(dim_1, dim_2)
    return sum([(dim + 1)-np.abs(i-dim) for i in range(0, whic)])

def arrays_to_diag(array_of_diags, dim_1, dim_2):
    full_lambda = []
    full_phi = []
    for i in range(0,dim_1 + 1): #these are rows
        column_lambda = []
        column_phi = []
        for j in range(0,dim_2 + 1): #these are columns
            list_index = min(dim_2-i, j) #could be dim 1
            which_list = j + i
            new_list_idx = list_index + index_func(which_list, dim_1, dim_2)
            column_lambda.append(array_of_diags[new_list_idx][0])
            column_phi.append(array_of_diags[new_list_idx][1])
        full_lambda.append(column_lambda)
        full_phi.append(column_phi)
    return np.array(full_lambda), np.array(full_phi)