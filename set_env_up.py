try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


numpy_include_dir = numpy.get_include()

#kd tree
pykdtree = Extension(
    'dmifnet.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'dmifnet/utils/libkdtree/pykdtree/kdtree.c',
        'dmifnet/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

# marching cubes algorithm
mcubes_module = Extension(
    'dmifnet.utils.libmcubes.mcubes',
    sources=[
        'dmifnet/utils/libmcubes/mcubes.pyx',
        'dmifnet/utils/libmcubes/pywrapper.cpp',
        'dmifnet/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'dmifnet.utils.libmesh.triangle_hash',
    sources=[
        'dmifnet/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)


mise_module = Extension(
    'dmifnet.utils.libmise.mise',
    sources=[
        'dmifnet/utils/libmise/mise.pyx'
    ],
)

simplify_mesh_module = Extension(
    'dmifnet.utils.libsimplify.simplify_mesh',
    sources=[
        'dmifnet/utils/libsimplify/simplify_mesh.pyx'
    ]
)

voxelize_module = Extension(
    'dmifnet.utils.libvoxelize.voxelize',
    sources=[
        'dmifnet/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)


dmc_pred2mesh_module = CppExtension(
    'dmifnet.dmc.ops.cpp_modules.pred2mesh',
    sources=[
        'dmifnet/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
    ]   
)

dmc_cuda_module = CUDAExtension(
    'dmifnet.dmc.ops._cuda_ext', 
    sources=[
        'dmifnet/dmc/ops/src/extension.cpp',
        'dmifnet/dmc/ops/src/curvature_constraint_kernel.cu',
        'dmifnet/dmc/ops/src/grid_pooling_kernel.cu',
        'dmifnet/dmc/ops/src/occupancy_to_topology_kernel.cu',
        'dmifnet/dmc/ops/src/occupancy_connectivity_kernel.cu',
        'dmifnet/dmc/ops/src/point_triangle_distance_kernel.cu',
    ]
)


ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    dmc_pred2mesh_module,
    dmc_cuda_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmdclass={
        'build_ext': BuildExtension
    }
)
