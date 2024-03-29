from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys
import setuptools

__version__ = '1.0.0'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    flags = ['-std=c++14']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': ['-O2', '-march=native'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['-fopenmp'],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    elif sys.platform == 'linux':
        linux_opts = ['-fopenmp']
        c_opts['unix'] += linux_opts
        l_opts['unix'] += linux_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        print(opts)

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        'movie_upsampling.upsampling_cpp_lib',
        ['movie_upsampling/upsampling_cpp_lib/movie_upsampling_pbind.cpp', ],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    CUDAExtension('movie_upsampling.torch_sparse_upsample_cuda', [
        'movie_upsampling/torch_sparse_upsample_cuda/upsample_cuda.cpp',
        'movie_upsampling/torch_sparse_upsample_cuda/upsampling.cu',
    ]),
    CUDAExtension('movie_upsampling.jitter_cuda', [
        'movie_upsampling/jitter_cuda/jitter_cuda.cpp',
        'movie_upsampling/jitter_cuda/jitter.cu',
    ]),
    CUDAExtension('movie_upsampling.diff_upsample', [
        'movie_upsampling/diff_upsample/diff_upsample_cuda.cpp',
        'movie_upsampling/diff_upsample/diff_upsample.cu',
    ])
]

py_modules = [
    'movie_upsampling',
]

setup(
    name='movie_upsampling',
    version=__version__,
    author='Eric Wu',
    ext_modules=ext_modules,
    packages=py_modules,
    install_requires=['pybind11>=2.3', 'numpy'],
    setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
