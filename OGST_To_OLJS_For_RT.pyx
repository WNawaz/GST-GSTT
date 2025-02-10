# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
# distutils: language = c
# cython: c_string_type=unicode, c_string_encoding=utf8
import os
import xarray as xr
import numpy as np
import networkit as nk
cimport numpy as cnp
cimport cython as cy
from cython.parallel import prange
from numpy import log2
from scipy import stats as sts
from sklearn.metrics.cluster import adjusted_rand_score as ars
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from SLAQ import netlsd, vnge
from scipy.sparse import csr_matrix as spsm
from time import time


@cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef similarity(str p_inpnet,
Py_ssize_t p_n,
Py_ssize_t p_edg_num,
cnp.float64_t p_scaling,
cnp.float64_t p_tolerance,
str p_method,
str p_oupnet,
Py_ssize_t p_repeat,
str p_suffix):

    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] Gp, Gp_Glo_lsd, pts
        cnp.ndarray[cnp.float32_t, ndim=1] Gp_Glo, Gp_Glo_vnge, ratio, gccr, lgccr, dcr, lccr, bcr, arsr, nmir, pdcr, plccr, pbcr, netlsdr, vnger, rpt, Q, cd, dc, lcc, bc
        cnp.ndarray[cnp.int32_t, ndim=2] g_edgw
        cnp.ndarray[cnp.int32_t, ndim=1] shift, g_deg, g_wdeg, g_cudeg, g_adj, g_wadj
        # for memoryview
        cnp.float32_t[:] g_lcc_v
        cnp.int32_t[:, :] g_edgw_v
        cnp.int32_t[:] shift_v, g_deg_v, g_wdeg_v, g_cudeg_v, g_adj_v, g_wadj_v, mk_v, g_ni_v, g_wni_v, g_nj_v
        # single variable
        Py_ssize_t rp=0, g_edg_num=0, e=0, i=0, SAMPLE=100, ii=0, j=0, jj=0
        cnp.int32_t eid=0, nd1=0, nd2=0, num=0, idx=0, gc3=0
        cnp.float32_t netlsdv=0, vngev=0, w=0
        str INP_g = ''
        object g, p, g_dc, g_bc, sps_net


    THREAD = 1
    nk.setNumberOfThreads(THREAD)


    Gp = np.load(f'{p_inpnet}_Gstats.npy').astype(np.float32)
    Gp_Glo = np.load(f'{p_inpnet}_Gstats_Global.npy').astype(np.float32)
    Gp_Glo_lsd = np.load(f'{p_inpnet}_Gstats_Spectral.npy',
                         allow_pickle=True)[0].astype(np.float32)
    Gp_Glo_vnge = np.load(f'{p_inpnet}_Gstats_Spectral.npy',
                          allow_pickle=True)[1][0].astype(np.float32)
    if p_suffix != 'No':
        OUP_gStats = f'{p_inpnet}_gljsstats_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_gStats_global = f'{p_inpnet}_gljsstats_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_gStats_global_netlsd = f'{p_inpnet}_gljsslaq_netlsd_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_gStats_global_vnge = f'{p_inpnet}_gljsslaq_vnge_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_RT = f'{p_inpnet}_gljs_rt{p_repeat}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
    else:
        OUP_gStats = f'{p_inpnet}_gljsstats_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_gStats_global = f'{p_inpnet}_gljsstats_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_gStats_global_netlsd = f'{p_inpnet}_gljsslaq_netlsd_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_gStats_global_vnge = f'{p_inpnet}_gljsslaq_vnge_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_RT = f'{p_inpnet}_gljs_rt{p_repeat}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'

    ratio = np.zeros(p_repeat, dtype=np.float32)
    gccr = np.zeros(p_repeat, dtype=np.float32)
    lgccr = np.zeros(p_repeat, dtype=np.float32)
    dcr = np.zeros(p_repeat, dtype=np.float32)
    lccr = np.zeros(p_repeat, dtype=np.float32)
    bcr = np.zeros(p_repeat, dtype=np.float32)
    arsr = np.zeros(p_repeat, dtype=np.float32)
    nmir = np.zeros(p_repeat, dtype=np.float32)

    pdcr = np.zeros(p_repeat, dtype=np.float32)
    plccr = np.zeros(p_repeat, dtype=np.float32)
    pbcr = np.zeros(p_repeat, dtype=np.float32)

    netlsdr = np.zeros(p_repeat, dtype=np.float32)
    vnger = np.zeros(p_repeat, dtype=np.float32)

    rpt = np.zeros(p_repeat, dtype=np.float32)


    # if not os.path.exists(OUP_gStats_global) or not os.path.exists(OUP_gStats) or not os.path.exists(OUP_RT):


    print(p_inpnet, p_n, p_edg_num, p_scaling, p_tolerance, p_method, p_oupnet, p_suffix)
    G = nk.readGraph(f'{p_inpnet}_wgph.graph', nk.Format.METIS)
    t1 = time()
    G.indexEdges()
    t2 = time() - t1


    for rp in range(p_repeat):
        if p_suffix != 'No':
            INP_g = f'{p_oupnet}_wgph_{str(rp)}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.graph'
        else:
            INP_g = f'{p_oupnet}_wgph_{str(rp)}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.graph'
        g_edg_num = nk.readGraph(INP_g, nk.Format.METIS).numberOfEdges()
        ratio[rp] = <float>(g_edg_num) / <float>(p_edg_num)

        
        # Sparsification ==================================
        t3 = time()
        g = nk.sparsification.LocalSimilaritySparsifier().getSparsifiedGraphOfSize(G, ratio[rp])
        rpt[rp] = time() - t3 + t2
        # Sparsification ==================================
    np.save(OUP_RT, np.array([np.mean(rpt), np.std(rpt)]))