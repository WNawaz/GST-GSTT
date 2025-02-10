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
        cnp.ndarray[cnp.int32_t, ndim=2] edgw
        cnp.ndarray[cnp.float64_t, ndim=1] Sptl_netlsd
        Py_ssize_t SAMPLE=100, REPEAT=10, THREAD=16, NODES=0
        list DATASETS_LIST, NODES_LIST, Sptl_vnge
        str DATASETS=''
        str OUP_GStats_Spectra=''
        object sps_Net, data


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
        OUP_GStats_Spectral = f'{DATASETS}_Gstats_Spectral.npy'

        # weighted Spectral done by SELF
        edgw = np.load(f'{DATASETS}_Udwp_1.npy')[:, :3].astype(np.int32) - 1
        sps_Net = spsm((np.tile(edgw[:, 2], 2).astype(np.float32) + 1,
                        (np.concatenate((edgw[:, 0], edgw[:, 1])),
                        np.concatenate((edgw[:, 1], edgw[:, 0])))),
                    shape=(NODES, NODES))
        Sptl_netlsd = np.empty(0)
        Sptl_vnge = []
        for i in range(REPEAT):
            Sptl_netlsd = np.concatenate((Sptl_netlsd, netlsd(sps_Net)))
            Sptl_vnge.append(vnge(sps_Net))
        data = np.array([Sptl_netlsd.reshape((REPEAT, -1)),
                        np.array([Sptl_vnge])],
                        dtype=object)
        np.save(OUP_GStats_Spectral, data)
