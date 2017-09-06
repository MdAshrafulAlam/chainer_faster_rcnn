from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = 'Non Maximum Suppression',
        ext_modules = cythonize('src/_nms_gpu_post.pyx'),
)
