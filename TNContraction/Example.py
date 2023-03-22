import numpy as np
import time

dim = 10
I = np.random.rand(dim, dim, dim, dim)
C = np.random.rand(dim, dim)

def naive(I, C):
    # N^8 scaling
    return np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)

def def_optim(I, C):
    # N^8 scaling
    return np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C, optimize=True)

def custom_optim(I, C):
    # N^5 scaling
    K = np.einsum('pi,ijkl->pjkl', C, I)
    K = np.einsum('qj,pjkl->pqkl', C, K)
    K = np.einsum('rk,pqkl->pqrl', C, K)
    K = np.einsum('sl,pqrl->pqrs', C, K)
    return K

if __name__ == "__main__":
    t0 = time.time()
    final_naive = naive(I,C)
    t1 = time.time()
    final_def_opti = def_optim(I,C)
    t2 = time.time()
    final_custom_optim = custom_optim(I,C)
    t3 = time.time()

    print("NAIVE TIME", t1-t0)
    print("OPTIMIZED TIME", t2 - t1)
    print("CUSTOM OPTIMIZED TIME", t3 - t2)


