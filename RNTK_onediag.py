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

        ns_dim_num = self.dim_1 + self.dim_2 + 1
        clip_num = min(self.dim_1, self.dim_2) + 1
        middle_list = np.zeros(ns_dim_num-(2 * clip_num) + 1)
        middle_list.fill(clip_num)
        self.dim_lengths = np.concatenate([np.arange(1,clip_num), middle_list, np.arange(clip_num, 0, -1)])

        self.how_many_before = [sum(self.dim_lengths[:j]) for j in range(0, len(self.dim_lengths))]

        length_betw = (self.dim_lengths - 1)[1:-1]
        self.when_to_reset = np.array([sum(length_betw[:j]) for j in range(0, len(length_betw)+1)])[1:] - 1

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

    def create_func_for_diag(self):

        bc = self.sh ** 2 * self.sw ** 2 * T.eye(self.n, self.n) + (self.su ** 2)* self.X + self.sb ** 2 ## took out X || 
        single_boundary_condition = T.expand_dims(bc, axis = 0)
        # single_boundary_condition = T.expand_dims(T.Variable((bc), "float32", "boundary_condition"), axis = 0)
        boundary_condition = T.concatenate([single_boundary_condition, single_boundary_condition])
        self.boundary_condition = boundary_condition

        ## prev_vals - (2,1) - previous phi and lambda values
        ## idx - where we are on the diagonal
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

            if idx in self.when_to_reset:
                return boundary_condition, to_return
            return to_return, to_return
        
        last_ema, all_ema = T.scan(
            fn, init =  boundary_condition, sequences=[jnp.arange(0, sum(self.dim_lengths) - self.dim_num)], non_sequences=[self.X]
        )
        return all_ema

    def diag_func_wrapper(self, dim_1_idx, dim_2_idx, fbool = False, jmode = False):
        # print('tests')
        f = self.create_func_for_diag(dim_1_idx, dim_2_idx, function = fbool, jmode = jmode)
        # print('teste')
        if fbool:
            return f(np.arange(0,min(self.dim_1, self.dim_2) - (dim_1_idx + dim_2_idx)), self.dim_1, self.dim_2, dim_1_idx, dim_2_idx, self.n)
        return f

    def no_bc_arrays_to_diag(self, input_array):

        indices_to_set = np.sort(list(set(range(0,int(sum(self.dim_lengths)))) - set(self.how_many_before)))
        array_of_diags = np.zeros(int(sum(self.dim_lengths)), dtype = "object")
        array_of_diags.fill(self.boundary_condition)
        np.put(array_of_diags, indices_to_set, [input_array[i] for i in range(input_array.shape[0])])

        full_lambda = []
        full_phi = []
        for i in range(0, self.dim_1 + 1): #these are rows
            column_lambda = []
            column_phi = []
            for j in range(0, self.dim_2 + 1): #these are columns
                list_index = min(self.dim_1-i, j) #could be dim 1
                which_list = j + i
                new_list_idx = list_index + int(self.how_many_before[which_list])
                column_lambda.append(array_of_diags[new_list_idx][0])
                column_phi.append(array_of_diags[new_list_idx][1])
            full_lambda.append(column_lambda)
            full_phi.append(column_phi)
        return np.array(full_lambda), np.array(full_phi)