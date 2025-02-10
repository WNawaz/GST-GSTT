# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import sys
import numpy as np
from build.Network import net as NETWORK


# network files ==================================
inpnet = "./Results/ERA5_SP/Spr_1998_To_2019_JJA_00_d0_P995950_X3"
n, m = 7320, 593736
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_2mTemperature/Spr_1998_To_2019_JJA_00_d0_P995950_X3"
n, m = 7320, 882102
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_GPH_250hpa/Spr_1998_To_2019_JJA_00_d0_P995950_X3"
n, m = 7320, 778757
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_OLR/Spr_1998_To_2019_JJA_00_d0_P950950_X3"
n, m = 7320, 422724
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_WUC_250hpa/Spr_1998_To_2019_JJA_00_d0_P950950_X3"
n, m = 7320, 541877
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_WVC_250hpa/Spr_1998_To_2019_JJA_00_d0_P950950_X3"
n, m = 7320, 340725
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_WVFUC/Spr_1998_To_2019_JJA_00_d0_P950950_X3"
n, m = 7320, 482762
NETWORK(inpnet, n, m)
inpnet = "./Results/ERA5_WVFVC/Spr_1998_To_2019_JJA_00_d0_P950950_X3"
n, m = 7320, 344020
NETWORK(inpnet, n, m)
inpnet = "./Results/TRMM_Precipitation/ES_1998_To_2019_JJA_900_d10_P990_X4"
n, m = 36000, 2139214
NETWORK(inpnet, n, m)
inpnet = "./Results/TRMM_Precipitation/ES_1998_To_2019ASM_JJA_900_d10_P990_X2"
n, m = 20000, 1771609
NETWORK(inpnet, n, m)

inpnet = "./Results/Real_Supp/Chameleon"
n, m = 2277, 31371
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/FBEgo"
n, m = 4039, 88234
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Crocodile"
n, m = 11631, 170773
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/HepPh"
n, m = 12008, 118489
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/AstroPh"
n, m = 18772, 198050
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/ASI"
n, m = 34761, 107720
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Enron"
n, m = 36692, 183831
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Livemocha"
n, m = 104103, 2193083
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Squirrel"
n, m = 5201, 198353
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Worm"
n, m = 16347, 762822
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Recmv"
n, m = 61989, 2811458
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Catster"
n, m = 149700, 5448197
NETWORK(inpnet, n, m)
inpnet = "./Results/Real_Supp/Twitch"
n, m = 168114, 6797557
NETWORK(inpnet, n, m)

inpnet = "./Results/LFR/10000_LFR_01"
n, m = 10000, 252039
NETWORK(inpnet, n, m)
inpnet = "./Results/LFR/10000_LFR_02"
n, m = 10000, 252916
NETWORK(inpnet, n, m)
inpnet = "./Results/LFR/10000_LFR_03"
n, m = 10000, 249388
NETWORK(inpnet, n, m)
inpnet = "./Results/LFR/10000_LFR_04"
n, m = 10000, 251048
NETWORK(inpnet, n, m)
print("Hello, my friend..........................")
