from setuptools import setup
from torch.utils import cpp_extension

cuda_max_perf_args = ['-O3', '--use_fast_math']
setup(name='online_softmax',
      ext_modules=[cpp_extension.CUDAExtension(
          name = 'online_softmax_cu', 
          sources = ['csrc/online_softmax.cu'],
      extra_compile_args={'cxx': ['-O3'],
                                'nvcc': cuda_max_perf_args}
      )],
      cmdclass={'build_ext': cpp_extension.BuildExtension})