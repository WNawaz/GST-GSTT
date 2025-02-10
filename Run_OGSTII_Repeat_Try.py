# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import sys
from build.OGSTI import expectation as OGST_STAGEI
from build.OGSTII import sparsification as OGST_STAGEII
from build.OGSTII_Repeat import sparsification as OGST_REPEAT


# inpnet, nodes, edges, scaling, suffix
# OGST_STAGEI(sys.argv[1],
#            int(sys.argv[2]),
#            int(sys.argv[3]),
#            float(sys.argv[4]),
#            sys.argv[5])
# inpnet, nodes, edges, scaling, torlerance, method, oupnet, suffix
# OGST_STAGEII('./Results/TRMM_Precipitation/ES_1998_To_2019_JJA_900_d10_P990_X3',
#             64320,
#             6503352,
#             0.1,
#             0.01,
#             'GMN23',
#             './Results/TRMM_Precipitation/ES_1998_To_2019_JJA_900_d10_P990_X3',
#             'No')
# OGST_STAGEII(sys.argv[1],
#             int(sys.argv[2]),
#             int(sys.argv[3]),
#             float(sys.argv[4]),
#             float(sys.argv[5]),
#             sys.argv[6],
#             sys.argv[7],
#             sys.argv[8])
OGST_REPEAT(sys.argv[1],
            int(sys.argv[2]),
            int(sys.argv[3]),
            float(sys.argv[4]),
            float(sys.argv[5]),
            sys.argv[6],
            sys.argv[7],
            int(sys.argv[8]),
            sys.argv[9])
print('Hello, my friend..........................')
