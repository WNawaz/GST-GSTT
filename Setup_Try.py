# -*- coding: utf-8 -*-
# The lower left point in grid cells is the 1st.
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "Network",
        ["Network.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "Original_Property_Spectral",
        ["Original_Property_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "Original_Property_Global_To_Local",
        ["Original_Property_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTI",
        ["OGSTI.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII",
        ["OGSTII.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII_Example",
        ["OGSTII_Example.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII_Repeat",
        ["OGSTII_Repeat.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII_Repeat_For_Fixing",
        ["OGSTII_Repeat_For_Fixing.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_Similarity_Global_To_Local",
        ["OGST_Similarity_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_Similarity_Spectral",
        ["OGST_Similarity_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_ORES_Similarity_Global_To_Local",
        ["OGST_To_ORES_Similarity_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_ORES_Similarity_Spectral",
        ["OGST_To_ORES_Similarity_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OCN_Similarity_Global_To_Local",
        ["OGST_To_OCN_Similarity_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OCN_Similarity_Spectral",
        ["OGST_To_OCN_Similarity_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OLD_Similarity_Global_To_Local",
        ["OGST_To_OLD_Similarity_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OLD_Similarity_Spectral",
        ["OGST_To_OLD_Similarity_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OLJS_Similarity_Global_To_Local",
        ["OGST_To_OLJS_Similarity_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OLJS_Similarity_Spectral",
        ["OGST_To_OLJS_Similarity_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGSTII_Repeat_For_RT",
        ["OGSTII_Repeat_For_RT.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OCN_For_RT",
        ["OGST_To_OCN_For_RT.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OLD_For_RT",
        ["OGST_To_OLD_For_RT.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_OLJS_For_RT",
        ["OGST_To_OLJS_For_RT.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_ORES_For_RT",
        ["OGST_To_ORES_For_RT.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_ORRES_Similarity_Global_To_Local",
        ["OGST_To_ORRES_Similarity_Global_To_Local.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_ORRES_Similarity_Spectral",
        ["OGST_To_ORRES_Similarity_Spectral.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
    Extension(
        "OGST_To_ORRES_For_RT",
        ["OGST_To_ORRES_For_RT.pyx"],
        include_dirs=[np.get_include()],
        library_dirs=[],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_24_3_API_VERSION")],
        # windows
        libraries=[],
        extra_compile_args=["/openmp"],
        #   extra_link_args=['/openmp'],
        # linux
        #   libraries=["m"],
        #   extra_compile_args=['-fopenmp'],
        #   extra_link_args=['-fopenmp'],
    ),
]
setup(
    name="Extreme_Evol_EAS",
    ext_modules=cythonize(
        extensions,
        language_level="3",
        build_dir="build",
    ),
)
