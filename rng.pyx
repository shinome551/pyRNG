# distutils: language=c++
# distutils: extra_compile_args = ["-O3"]
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True

cimport numpy as np
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.pair cimport pair
from libc.float cimport FLT_MAX


ctypedef unsigned int TY
ctypedef np.float32_t DTYPE_t


cdef TY getIndexfromTriuVec(TY i, TY j, TY n, TY cumsum_n):
    if j > i:
        return cumsum_n -  (n - i) * (n - i - 1) / 2 + j - i - 1
    else:
        return cumsum_n - (n - j) * (n - j - 1) / 2 + i - j - 1


cpdef vector[float] calcDistanceTriuMatrix(vector[vector[float]] data):
    cdef:
        TY n = data.size()
        TY d = data[0].size()
        vector[float] dist_mat_triu
        TY i, j, k
        float sum
        
    for i in range(n):
        for j in range(i+1, n):
            sum = 0
            for k in range(d):
                sum += (data[i][k] - data[j][k]) ** 2
            dist_mat_triu.push_back(sum ** 0.5)

    return dist_mat_triu


cpdef vector[pair[TY, TY]] buildRelativeNeighborhoodGraph(DTYPE_t[:] dist_mat_triu, TY n):
    cdef:
        TY i, j, k, argmin, m, cumsum_n, n_cand
        float dist_ij, dist_ik, dist_jk, dist_mk, vmin, dist
        bool lune_empty
        vector[bool] rng    
        vector[TY] cand
        vector[bool] black
        queue[TY] open
    
    cand.reserve(n)
    cumsum_n = n * (n - 1) / 2
    rng.reserve(cumsum_n)
    black.reserve(cumsum_n)
    
    for i in range(n):
        open.push(i)
        
    for i in range(cumsum_n):
        black.push_back(False)
        rng.push_back(False)

    while not open.empty():
        i = open.front()
        open.pop()
        cand.clear()
        for j in range(n):
            if i == j:
                continue
            if not black[getIndexfromTriuVec(i, j, n, cumsum_n)]:
                cand.push_back(j)
        if not cand.empty():
            vmin, argmin = FLT_MAX, 0
            n_cand = cand.size()
            for j in range(n_cand):
                dist = dist_mat_triu[getIndexfromTriuVec(i, cand[j], n, cumsum_n)]
                if vmin > dist:
                    vmin, argmin = dist, cand[j]
            
            dist_ij, j = vmin, argmin
            black[getIndexfromTriuVec(i, j, n, cumsum_n)] = True

            lune_empty = True
            for k in range(n):
                if k == i or k == j:
                    continue
                dist_ik = dist_mat_triu[getIndexfromTriuVec(i, k, n, cumsum_n)]
                dist_jk = dist_mat_triu[getIndexfromTriuVec(j, k, n, cumsum_n)]
                if dist_ik > dist_jk:
                    dist_mk, m = dist_ik, i
                else:
                    dist_mk, m = dist_jk, j
                if dist_mk < dist_ij:
                    lune_empty = False
                else:
                    black[getIndexfromTriuVec(m, k, n, cumsum_n)] = True
            rng[getIndexfromTriuVec(i, j, n, cumsum_n)] = lune_empty
            open.push(i)

    return getEdges(rng, n)


cdef vector[pair[TY, TY]] getEdges(vector[bool] isEdge, TY n):
    cdef:
        TY i, j, k, n_edges
        vector[pair[TY, TY]] edges
        pair[TY, TY] p
    
    n_edges = 0
    for i in range(n * (n - 1) / 2):
        n_edges += isEdge[i]
    edges.reserve(n_edges)
    
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            if isEdge[k]:
                p.first = i
                p.second = j
                edges.push_back(p)
            k += 1
    return edges