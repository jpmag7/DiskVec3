from setuptools import setup, Extension
import pybind11
import sys

# Define platform-specific settings
is_windows = sys.platform.startswith('win')

# Base compiler flags
extra_compile_args = []
define_macros = [('BUILD_PYBINDINGS', None)]

if is_windows:
    # Windows-specific settings
    extra_compile_args.extend(['/O3'])
    # Tell the compiler we're on Windows
    define_macros.append(('_WIN32', None))
else:
    # Unix-specific settings
    extra_compile_args.extend(['-O3', '-std=c++11'])

module = Extension(
    'diskvec',
    sources=['diskvec.cpp'],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
)

setup(
    name='diskvec',
    version='0.1',
    description='VP Tree module with in-place reordering',
    ext_modules=[module]
)