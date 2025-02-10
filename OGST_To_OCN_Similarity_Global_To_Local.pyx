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
        cnp.ndarray[cnp.float32_t, ndim=2] Gp, Gp_Glo_lsd, pts, G_es
        cnp.ndarray[cnp.float32_t, ndim=1] Gp_Glo, Gp_Glo_vnge, ratio, gccr, lgccr, dcr, lccr, bcr, arsr, nmir, pdcr, plccr, pbcr, netlsdr, vnger, rpt, Es, Q, cd, dc, lcc, bc
        cnp.ndarray[cnp.int32_t, ndim=2] g_edgw
        cnp.ndarray[cnp.int32_t, ndim=1] G_deg, G_cudeg, G_adj, shift, idx_g_edge, g_deg, g_wdeg, g_cudeg, g_adj, g_wadj
        cnp.ndarray[cnp.int8_t, ndim=1] mk3, mkG
        # for memoryview
        cnp.float32_t[:, :] G_es_v
        cnp.float32_t[:] g_lcc_v
        cnp.int32_t[:, :] g_edgw_v
        cnp.int32_t[:] G_deg_v, G_cudeg_v, G_adj_v, shift_v, Gnn1_v, Gwn1_v, Gnn2_v, g_deg_v, g_wdeg_v, g_cudeg_v, g_adj_v, g_wadj_v, mk_v, g_ni_v, g_wni_v, g_nj_v
        cnp.int8_t[:] mk3_v, mkG_v
        # single variable
        Py_ssize_t rp=0, g_edg_num=0, e=0, i=0, ni=0, mk=0, nj=0, SAMPLE=100, ii=0, j=0, jj=0
        cnp.int32_t eid=0, idx=0, nd1=0, nd2=0, wn_gnn12=0, num=0, gc3=0
        cnp.float32_t netlsdv=0, vngev=0
        str INP_DEG = ''
        str INP_CUDEG = ''
        str INP_ADJ = ''
        str INP_g = ''
        object g, p, g_dc, g_bc, sps_net


    THREAD = 16
    nk.setNumberOfThreads(THREAD)


    Gp = np.load(f'{p_inpnet}_Gstats.npy').astype(np.float32)
    Gp_Glo = np.load(f'{p_inpnet}_Gstats_Global.npy').astype(np.float32)
    Gp_Glo_lsd = np.load(f'{p_inpnet}_Gstats_Spectral.npy',
                         allow_pickle=True)[0].astype(np.float32)
    Gp_Glo_vnge = np.load(f'{p_inpnet}_Gstats_Spectral.npy',
                          allow_pickle=True)[1][0].astype(np.float32)
    if p_suffix != 'No':
        OUP_gStats = f'{p_inpnet}_gcnstats_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_gStats_global = f'{p_inpnet}_gcnstats_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_gStats_global_netlsd = f'{p_inpnet}_gcnslaq_netlsd_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_gStats_global_vnge = f'{p_inpnet}_gcnslaq_vnge_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
        OUP_RT = f'{p_inpnet}_gcn_rt{p_repeat}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO_{p_suffix}.npy'
    else:
        OUP_gStats = f'{p_inpnet}_gcnstats_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_gStats_global = f'{p_inpnet}_gcnstats_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_gStats_global_netlsd = f'{p_inpnet}_gcnslaq_netlsd_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_gStats_global_vnge = f'{p_inpnet}_gcnslaq_vnge_global_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'
        OUP_RT = f'{p_inpnet}_gcn_rt{p_repeat}_{p_method}_{str(p_tolerance).split(".", 1)[0] + str(p_tolerance).split(".", 1)[1]}_S{str(p_scaling).split(".", 1)[0] + str(p_scaling).split(".", 1)[1]}_CyO.npy'

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

    t1 = time()
    print(p_inpnet, p_n, p_edg_num, p_scaling, p_tolerance, p_method, p_oupnet, p_suffix)
    INP_DEG = f'{p_inpnet}_deg.npy'
    INP_CUDEG = f'{p_inpnet}_cudeg.npy'
    INP_ADJ = f'{p_inpnet}_adj.npy'
    G_deg = np.load(INP_DEG).astype(np.int32)
    G_cudeg = np.load(INP_CUDEG).astype(np.int32)
    G_adj = np.load(INP_ADJ).astype(np.int32)
    G_deg_v = G_deg
    G_cudeg_v = G_cudeg
    G_adj_v = G_adj
    G_es = np.zeros((p_edg_num, 3), dtype=np.float32)
    mk3 = np.zeros(p_n, dtype=np.int8)
    mkG = np.ones(int(2 * p_edg_num), dtype=np.int8)
    G_es_v = G_es
    mk3_v = mk3
    mkG_v = mkG
    eid = 0
    for i in range(p_n):
        if G_deg_v[i] != 0:
            for ni in range(G_deg_v[i]):      
                Gnn1_v = G_adj_v[G_cudeg_v[i]:G_cudeg_v[i + 1]]
                if ni == 0:  # reduce computational cost
                    for mk in range(G_deg_v[i]):
                        mk3_v[Gnn1_v[mk]] = 1

                # to make sure each round, iterating each edge only once
                if mkG_v[G_cudeg_v[i] + ni] == 1:
                    j = Gnn1_v[ni]
                    Gnn2_v = G_adj_v[G_cudeg_v[j]:G_cudeg_v[j + 1]]

                    mkG_v[G_cudeg_v[i] + ni] = 0
                    for mk in range(G_deg_v[j]):
                        if Gnn2_v[mk] == i:
                            mkG_v[G_cudeg_v[j] + mk] = 0
                            break
                    wn_gnn12 = 0
                    for nj in range(G_deg_v[j]):
                        if mk3_v[Gnn2_v[nj]] == 1:
                            wn_gnn12 += 1
                    G_es_v[eid, 0] = <float>(i)
                    G_es_v[eid, 1] = <float>(j)
                    G_es_v[eid, 2] = <float>(wn_gnn12)
                    eid += 1
                else:
                    continue
            for mk in range(G_deg_v[i]):
                mk3_v[Gnn1_v[mk]] = 0
    if eid != p_edg_num:
        print("Edge list length not matching!!!!!!!")
        os._exit(0)
    del G_adj
    G_es = G_es[np.lexsort((G_es[:, 1], G_es[:, 0])), :]
    # print(np.array_equal(G_es[:, :2].astype(np.int32),
    #                      np.load(f'{p_inpnet}_Udwp_1.npy')[:, :2].astype(np.int32) - 1))
    # os._exit(0)
    Es = (2 / ((G_es[:, 2] + 2) * np.sum(2 / (G_es[:, 2]  + 2)))).astype(np.float32)
    idx_g_edge = np.argsort(Es)[::-1].astype(np.int32)
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
        g = nk.Graph(n=p_n, weighted=True)
        g_edgw = np.load(f'{p_inpnet}_Udwp_1.npy')[np.sort(idx_g_edge[:g_edg_num]).astype(np.int32), :].astype(np.int32) - 1
        g_edgw_v = g_edgw
        rpt[rp] = time() - t3 + t2
        # Sparsification ==================================


        # Query by NK =====================================
        # largest CC done by NK
        for e in range(g_edg_num):
            g.setWeight(g_edgw_v[e, 0],
                        g_edgw_v[e, 1],
                        g_edgw_v[e, 2] + 1)
        lgccr[rp] = abs(nk.components.ConnectedComponents.extractLargestConnectedComponent(
            g, True).numberOfNodes() - Gp_Glo[1]) / Gp_Glo[1]
        # community structure done by NK
        pts = np.zeros((SAMPLE, p_n), dtype=np.float32)
        Q = np.zeros(SAMPLE, dtype=np.float32)
        for i in range(SAMPLE):
            p = nk.community.detectCommunities(
                g, algo=nk.community.PLM(g, True, 1))
            pts[i, :] = np.array(
                p.getVector(), dtype=np.float32)
            Q[i] = nk.community.Modularity().getQuality(p, g)
        cd = pts[np.argsort(Q)[::-1]][0, :]
        arsr[rp] = ars(Gp[0, :], cd)
        nmir[rp] = nmi(Gp[0, :], cd)
        print("The first 2 qualities after ranking: \n",
            Q[np.argsort(Q)[::-1]][:2])
        # degree centrality done by NK
        g_dc = nk.centrality.DegreeCentrality(g)
        g_dc.run()
        dc = np.array(g_dc.scores()).astype(np.float32)
        if np.sum(dc) == 0:
            dcr[rp], pdcr[rp] = np.nan, np.nan
        else:
            dcr[rp], pdcr[rp] = sts.spearmanr(Gp[1, :], dc)
        # betweenness centrality done by NK
        # Running stops with KadabraBetweenness(g, err=0.05, delta=0.8)
        # g_bc = nk.centrality.Betweenness(g, normalized=True)
        # g_bc = nk.centrality.ApproxBetweenness(G, epsilon=0.01, delta=0.1)
        g_bc = nk.centrality.EstimateBetweenness(g, SAMPLE, True, True)
        bc = np.zeros(p_n, dtype=np.float32)
        for i in range(p_repeat):
            g_bc.run()
            bc += np.array(g_bc.scores(), dtype=np.float32)
        bc = bc / p_repeat
        if np.sum(np.mean(bc, axis=0)) == 0:
            bcr[rp], pbcr[rp] = np.nan, np.nan
        else:
            bcr[rp], pbcr[rp] = sts.spearmanr(Gp[3, :], bc)


        # Query by SELF ===================================
        g_deg = np.zeros(p_n, dtype=np.int32)
        g_wdeg = np.zeros(p_n, dtype=np.int32)
        g_deg_v = g_deg
        g_wdeg_v = g_wdeg
        for e in range(g_edg_num):
            nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
            g_deg_v[nd1] += 1
            g_deg_v[nd2] += 1
            g_wdeg_v[nd1] += (g_edgw_v[e][2] + 1)
            g_wdeg_v[nd2] += (g_edgw_v[e][2] + 1)
        g_cudeg = np.zeros(p_n + 1, dtype=np.int32)
        g_cudeg_v = g_cudeg
        num = 0
        for i in range(p_n + 1):
            g_cudeg_v[i] = num
            if i < p_n:
                num += g_deg_v[i]
        shift = g_cudeg.copy()
        shift_v = shift
        g_adj = np.zeros(np.sum(g_deg), dtype=np.int32)
        g_wadj = np.zeros(np.sum(g_deg), dtype=np.int32)
        g_adj_v = g_adj
        g_wadj_v = g_wadj
        for e in range(g_edg_num):
            nd1, nd2 = g_edgw_v[e][0], g_edgw_v[e][1]
            g_adj_v[shift_v[nd1]] = nd2
            g_adj_v[shift_v[nd2]] = nd1
            g_wadj_v[shift_v[nd1]] = (g_edgw_v[e][2] + 1)
            g_wadj_v[shift_v[nd2]] = (g_edgw_v[e][2] + 1)
            shift_v[nd1] += 1
            shift_v[nd2] += 1
        del g_edgw
        # weighted local clustering coefficient done by SELF
        lcc = np.zeros(p_n, dtype=np.float32)
        g_lcc_v = lcc
        mk_v = np.zeros(p_n, dtype=np.int32) - 1
        for i in range(p_n):
            if g_deg_v[i] > 1:
                g_ni_v = g_adj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
                g_wni_v = g_wadj_v[g_cudeg_v[i]:g_cudeg_v[i + 1]]
                idx = 0
                for ii in range(g_deg_v[i]):
                    mk_v[g_ni_v[ii]] = idx
                    idx += 1
                gc3 = 0
                for ii in range(g_deg_v[i] - 1):
                    j = g_ni_v[ii]
                    g_nj_v = g_adj_v[g_cudeg_v[j]:g_cudeg_v[j + 1]]
                    for jj in range(g_deg_v[j]):
                        if mk_v[g_nj_v[jj]] != -1:
                            idx = mk_v[g_nj_v[jj]]
                            gc3 += (g_wni_v[ii] + g_wni_v[idx])
                    mk_v[j] = -1
                mk_v[g_ni_v[ii + 1]] = -1
                g_lcc_v[i] = <float>((0.5 * gc3) / <float>(0.5 * g_wdeg_v[i] * (g_deg_v[i] - 1)))
        if np.sum(lcc) == 0:
            lccr[rp], plccr[rp] = np.nan, np.nan
        else:
            lccr[rp], plccr[rp] = sts.spearmanr(Gp[2, :], lcc)
        # weighted global clustering coefficient done by SELF
        if np.sum(lcc) == 0:
            gccr[rp] = 1
        else:
            gccr[rp] = abs(np.mean(lcc) - Gp_Glo[0]) / Gp_Glo[0]
        del g_adj, g_wadj
        # weighted Spectral done by SELF
        # sps_net = spsm((np.tile(g_edgw[:, 2], 2).astype(np.float32) + 1,
        #                 (np.concatenate((g_edgw[:, 0], g_edgw[:, 1])),
        #                  np.concatenate((g_edgw[:, 1], g_edgw[:, 0])))),
        #                shape=(p_n, p_n))
        # netlsdv = 0
        # vngev = 0
        # for i in range(p_repeat):
        #     netlsdv += np.linalg.norm(netlsd(sps_net) - Gp_Glo_lsd[i, :])
        #     vngev += np.linalg.norm(vnge(sps_net) - Gp_Glo_vnge[i])
        # netlsdr[rp] = <float>(netlsdv / p_repeat)
        # vnger[rp] = <float>(vngev / p_repeat)


    # Saving ==========================================
    np.save(OUP_gStats_global, np.array([[np.mean(ratio), np.std(ratio)],
                                        [np.mean(gccr), np.std(gccr)],
                                        [np.mean(lgccr), np.std(lgccr)]]))
    np.save(OUP_gStats, np.array([[np.mean(ratio), np.std(ratio)],
                                [np.mean(dcr), np.std(dcr)],
                                [np.mean(pdcr), np.std(pdcr)],
                                [np.mean(lccr), np.std(lccr)],
                                [np.mean(plccr), np.std(plccr)],
                                [np.mean(bcr), np.std(bcr)],
                                [np.mean(pbcr), np.std(pbcr)],
                                [np.mean(arsr), np.std(arsr)],
                                [np.mean(nmir), np.std(nmir)]]))
    # np.save(OUP_gStats_global_netlsd, np.array([[np.mean(ratio), np.std(ratio)],
    #                                             [np.mean(netlsdr), np.std(netlsdr)]]))
    # np.save(OUP_gStats_global_vnge, np.array([[np.mean(ratio), np.std(ratio)],
    #                                           [np.mean(vnger), np.std(vnger)]]))
    np.save(OUP_RT, np.array([np.mean(rpt), np.std(rpt)]))