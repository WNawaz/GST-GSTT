# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
# distutils: language = c
# cython: c_string_type=unicode, c_string_encoding=utf8
import os
import numpy as np
import networkit as nk
cimport numpy as cnp
cimport cython as cy
from SLAQ import netlsd, vnge
from scipy.sparse import csr_matrix as spsm



# @cy.boundscheck(False)
@cy.wraparound(False)
@cy.initializedcheck(False)
@cy.nonecheck(False)
@cy.cdivision(True)
@cy.profile(True)
cpdef network_property():

    cdef:
        cnp.ndarray[cnp.float32_t, ndim=2] pts
        cnp.ndarray[cnp.float32_t, ndim=1] Q, cd, dc, bc, lcc
        cnp.int32_t[:] G_deg, G_wdeg, G_cudeg, G_adj, G_wadj, mk, G_ni, G_wni, G_nj
        cnp.int32_t idx=0, gc3=0
        cnp.float32_t gcc=0
        Py_ssize_t SAMPLE=100, REPEAT=10, THREAD=16, NODES=0, lgcc=0, i=0, ii=0, j=0, jj=0
        list DATASETS_LIST, NODES_LIST, Sptl_vnge
        str DATASETS=''
        str OUP_GStats_Spectra=''
        str INP_DEG=''
        str INP_WDEG=''
        str INP_CUDEG=''
        str INP_ADJ=''
        str INP_WADJ=''
        str OUP_GStats=''
        str OUP_GStats_GLOBAL=''
        object G, p, G_dc, G_bc, sps_Net, data


    nk.setNumberOfThreads(THREAD)
    DATASETS_LIST = [
        "./Results/ERA5_SP/Spr_1998_To_2019_JJA_00_d0_P995950_X3",
        "./Results/ERA5_2mTemperature/Spr_1998_To_2019_JJA_00_d0_P995950_X3",
        "./Results/ERA5_GPH_250hpa/Spr_1998_To_2019_JJA_00_d0_P995950_X3",
        "./Results/ERA5_OLR/Spr_1998_To_2019_JJA_00_d0_P950950_X3",
        "./Results/ERA5_WUC_250hpa/Spr_1998_To_2019_JJA_00_d0_P950950_X3",
        "./Results/ERA5_WVC_250hpa/Spr_1998_To_2019_JJA_00_d0_P950950_X3",
        "./Results/ERA5_WVFUC/Spr_1998_To_2019_JJA_00_d0_P950950_X3",
        "./Results/ERA5_WVFVC/Spr_1998_To_2019_JJA_00_d0_P950950_X3",
        "./Results/TRMM_Precipitation/ES_1998_To_2019_JJA_900_d10_P990_X4",
        "./Results/TRMM_Precipitation/ES_1998_To_2019ASM_JJA_900_d10_P990_X2",
        "./Results/Real_Supp/Chameleon",
        "./Results/Real_Supp/FBEgo",
        "./Results/Real_Supp/Crocodile",
        "./Results/Real_Supp/HepPh",
        "./Results/Real_Supp/AstroPh",
        "./Results/Real_Supp/ASI",
        "./Results/Real_Supp/Enron",
        "./Results/Real_Supp/Livemocha",
        "./Results/Real_Supp/Squirrel",
        "./Results/Real_Supp/Worm",
        "./Results/Real_Supp/Recmv",
        "./Results/Real_Supp/Catster",
        "./Results/Real_Supp/Twitch",
        "./Results/LFR/10000_LFR_01",
        "./Results/LFR/10000_LFR_02",
        "./Results/LFR/10000_LFR_03",
        "./Results/LFR/10000_LFR_04"]
    NODES_LIST = [7320,
                  7320,
                  7320,
                  7320,
                  7320,
                  7320,
                  7320,
                  7320,
                  36000,
                  20000,
                  2277,
                  4039,
                  11631,
                  12008,
                  18772,
                  34761,
                  36692,
                  104103,
                  5201,
                  16347,
                  61989,
                  149700,
                  168114,
                  10000,
                  10000,
                  10000,
                  10000]

    for DATASETS, NODES in zip(DATASETS_LIST, NODES_LIST):
        print(DATASETS, NODES)
        INP_DEG = f'{DATASETS}_deg.npy'
        INP_WDEG = f'{DATASETS}_wdeg.npy'
        INP_CUDEG = f'{DATASETS}_cudeg.npy'
        INP_ADJ = f'{DATASETS}_adj.npy'
        INP_WADJ = f'{DATASETS}_wadj.npy'

        OUP_GStats = f'{DATASETS}_Gstats.npy'
        OUP_GStats_GLOBAL = f'{DATASETS}_Gstats_Global.npy'

        G = nk.readGraph(f'{DATASETS}_gph.graph', nk.Format.METIS)

        # Query by NK =====================================
        # largest CC done by NK
        lgcc = nk.components.ConnectedComponents.extractLargestConnectedComponent(
            G, True).numberOfNodes()
        # community structure done by NK
        if 'LFR' in DATASETS:
            cd = np.load(f'{DATASETS}_Partition.npy')
        else:
            pts = np.zeros((SAMPLE, NODES), dtype=np.float32)
            Q = np.zeros(SAMPLE, dtype=np.float32)
            for i in range(SAMPLE):
                p = nk.community.detectCommunities(
                    G, algo=nk.community.PLM(G, True, 1))
                pts[i, :] = np.array(p.getVector(), dtype=np.float32)
                Q[i] = nk.community.Modularity().getQuality(p, G)
            cd = pts[np.argsort(Q)[::-1]][0, :]
            print("The first 2 qualities after ranking: \n",
                Q[np.argsort(Q)[::-1]][:2])
        # degree centrality done by NK
        G_dc = nk.centrality.DegreeCentrality(G)
        G_dc.run()
        dc = np.array(G_dc.scores(), dtype=np.float32)
        # betweenness centrality done by NK
        G_bc = nk.centrality.EstimateBetweenness(G, SAMPLE, True, True)
        bc = np.zeros(NODES, dtype=np.float32)
        for i in range(REPEAT):
            G_bc.run()
            bc += np.array(G_bc.scores(), dtype=np.float32)
        bc = bc / REPEAT

        # Query by SELF ===================================
        G_deg = np.load(INP_DEG).astype(np.int32)
        G_wdeg = np.load(INP_WDEG).astype(np.int32)
        G_cudeg = np.load(INP_CUDEG).astype(np.int32)
        G_adj = np.load(INP_ADJ).astype(np.int32)
        G_wadj = np.load(INP_WADJ).astype(np.int32)
        # weighted local clustering coefficient done by SELF
        lcc = np.zeros(NODES, dtype=np.float32)
        mk = np.zeros(NODES, dtype=np.int32) - 1
        for i in range(NODES):
            if G_deg[i] > 1:
                G_ni = G_adj[G_cudeg[i]:G_cudeg[i + 1]]
                G_wni = G_wadj[G_cudeg[i]:G_cudeg[i + 1]]
                idx = 0
                for ii in range(G_deg[i]):
                    mk[G_ni[ii]] = idx
                    idx += 1
                gc3 = 0
                for ii in range(G_deg[i] - 1):
                    j = G_ni[ii]
                    G_nj = G_adj[G_cudeg[j]:G_cudeg[j + 1]]
                    for jj in range(G_deg[j]):
                        if mk[G_nj[jj]] != -1:
                            idx = mk[G_nj[jj]]
                            gc3 += (G_wni[ii] + G_wni[idx])
                    mk[j] = -1
                mk[G_ni[ii + 1]] = -1
                lcc[i] = <float>((0.5 * gc3) / (0.5 * G_wdeg[i] * (G_deg[i] - 1)))
        # weighted global clustering coefficient done by SELF
        gcc = <float>(np.mean(lcc))

        # print(dc)
        # print(G_deg)
        # print()
        # G_bc = nk.centrality.Betweenness(G, True)
        # G_bc.run()
        # print(bc)
        # print(np.array(G_bc.scores(), dtype=np.float32))
        # print()
        # G_lcc = nk.centrality.LocalClusteringCoefficient(G)
        # G_lcc.run()
        # print(lcc)
        # print(np.array(G_lcc.scores(), dtype=np.float32))
        # print()
        # print(gcc)
        # print(nk.globals.clustering(G))
        # print()

        np.save(OUP_GStats_GLOBAL, np.array([gcc, lgcc],
                                            dtype=np.float32))
        np.save(OUP_GStats, np.array([cd, dc, lcc, bc],
                                    dtype=np.float32))
