import numpy as np
import jax
import symjax
import symjax.tensor as T
import jax.numpy as jnp

class RNTK():
    def __init__(self, dic, dim_1, dim_2, X, n):
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.dim_num = self.dim_1 + self.dim_2 + 1
        self.sw = 1
        self.su = 1
        self.sb = 1
        self.sh = 1
        self.L = 1
        self.Lf = 0
        self.sv = 1
        self.X = X
        self.n = n
        self.N = int(dic["n_patrons1="])
        self.length = int(dic["n_entradas="])

    def get_diag_indices(self, jnpbool = False):
        switch_flag = 1
        dim_1_i = self.dim_1
        dim_2_i = 0

        tiprimes = []
        tis = []

        print(dim_1_i, dim_2_i, switch_flag)
        for d in range(0,self.dim_num):
            tiprime = dim_1_i
            ti = dim_2_i

            # diag_func(tiprime, ti, dim_1, dim_2, rntk)
            tiprimes.append(tiprime)
            tis.append(ti)

            if dim_1_i == 0:
                switch_flag -= 1
            else:
                dim_1_i = dim_1_i - 1
            if switch_flag <= 0:
                dim_2_i = dim_2_i + 1
        if jnpbool:
            return jnp.array(tiprimes), jnp.array(tis)    
        return np.array(tiprimes), np.array(tis)

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

    def make_inputs(self, dim1idx, dim2idx, jmode = False):
        # print('make_inputs')
        # print(dim1idx)
        # print(dim1idx + dim2idx)
        # print(jnp.min(jnp.array([self.dim_1, self.dim_2])))
        # test = jnp.min(jnp.array([self.dim_1, self.dim_2])) - (dim1idx + dim2idx)
        # print(T.arange(test))
        # print(jmode)
        if jmode:
            diagindex = jax.numpy.arange(0,jnp.min(jnp.array([self.dim_1, self.dim_2])) - (dim1idx + dim2idx))
        else:
            diagindex = jax.numpy.arange(0,min(self.dim_1, self.dim_2) - (dim1idx + dim2idx))
        # print('make_inputs m1')
        diag = T.Variable((diagindex), "float32", "dimension internal index")
        dim1ph = T.Variable(self.dim_1, "float32", "dimension 1 max")
        dim2ph = T.Variable(self.dim_2, "float32", "dimension 2 max")
        # print('make_inputs m2')
        dim1idxph = T.Variable(dim1idx, "float32", "dimension 1 index")
        dim2idxph = T.Variable(dim2idx, "float32", "dimension 2 index")
        nph = T.Variable(self.n, "float32", "n")
        # print('make_inputs end')
        return diag, dim1ph, dim2ph, dim1idxph, dim2idxph, nph

    def create_func_for_diag(self, dim1idx, dim2idx, function = False, jmode = False):
        diag, dim1ph, dim2ph, dim1idxph, dim2idxph, nph = self.make_inputs(dim1idx, dim2idx, jmode = jmode)
        # print('test')

        ## prev_vals - (2,1) - previous phi and lambda values
        ## idx - where we are on the diagonal
        ## d1idx - y value of first dimension diag start
        ## d2idx - x value of second dimension diag start
        ## d1ph - max value of first dimension
        ## d2ph - max value of second dimension
        bc = self.sh ** 2 * self.sw ** 2 * T.eye(self.n, self.n) + (self.su ** 2)* self.X + self.sb ** 2
        single_boundary_condition = T.expand_dims(bc, axis = 0)
        # single_boundary_condition = T.expand_dims(T.Variable((bc), "float32", "boundary_condition"), axis = 0)
        boundary_condition = T.concatenate([single_boundary_condition, single_boundary_condition]) #one for phi and lambda

        def fn(prev_vals, idx, Xph):
            # tiprime_iter = d1idx + idx
            # ti_iter = d2idx + idx
            prev_lambda = prev_vals[0]
            prev_phi = prev_vals[1]
            ## not boundary condition
            S, D = self.VT(prev_lambda)
            new_lambda = self.sw ** 2 * S + self.su ** 2 * Xph + self.sb ** 2 ## took out an X
            new_phi = new_lambda + self.sw ** 2 * prev_phi * D
            lambda_expanded = T.expand_dims(new_lambda, axis = 0)
            phi_expanded = T.expand_dims(new_phi, axis = 0)
            to_return = T.concatenate([lambda_expanded, phi_expanded])
            
            return to_return, to_return

        last_ema, all_ema = T.scan(
            fn, init = boundary_condition, sequences=[diag], non_sequences=[self.X]
        )
        # print('end')
        expanded_ema = T.concatenate([T.expand_dims(boundary_condition, axis = 0), all_ema])
        if function: 
            f = symjax.function(diag, dim1ph, dim2ph, dim1idxph, dim2idxph, nph, outputs=expanded_ema)
            return f
        else:
            return expanded_ema

    def diag_func_wrapper(self, dim_1_idx, dim_2_idx, fbool = False, jmode = False):
        # print('tests')
        f = self.create_func_for_diag(dim_1_idx, dim_2_idx, function = fbool, jmode = jmode)
        # print('teste')
        if fbool:
            return f(np.arange(0,min(self.dim_1, self.dim_2) - (dim_1_idx + dim_2_idx)), self.dim_1, self.dim_2, dim_1_idx, dim_2_idx, self.n)
        return f

    def index_func(self, whic, dim_1, dim_2):
        dim = min(dim_1, dim_2)
        return sum([(dim + 1)-np.abs(i-dim) for i in range(0, whic)])

    def arrays_to_diag(self, array_of_diags):
        ns_dim_num = self.dim_1 + self.dim_2 + 1
        clip_num = min(self.dim_1, self.dim_2) + 1
        middle_list = np.zeros(ns_dim_num-(2 * clip_num) + 1)
        middle_list.fill(clip_num)
        new_list = np.concatenate([np.arange(1,clip_num), middle_list, np.arange(clip_num, 0, -1)])

        how_many_before = [sum(new_list[:j]) for j in range(0, len(new_list))]

        full_lambda = []
        full_phi = []
        for i in range(0, self.dim_1 + 1): #these are rows
            column_lambda = []
            column_phi = []
            for j in range(0, self.dim_2 + 1): #these are columns
                list_index = min(self.dim_1-i, j) #could be dim 1
                which_list = j + i
                new_list_idx = list_index + int(how_many_before[which_list])
                column_lambda.append(array_of_diags[new_list_idx][0])
                column_phi.append(array_of_diags[new_list_idx][1])
            full_lambda.append(column_lambda)
            full_phi.append(column_phi)
        return np.array(full_lambda), np.array(full_phi)