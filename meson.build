project(
    'werpy',
    'c', 'cython',
    version : '3.3.0',
    license : 'BSD-3-Clause',
    meson_version : '>= 1.1.0',
    default_options : [
        'buildtype=release',
        'b_ndebug=if-release',
        'warning_level=3',
        'c_std=c17'
        ]
)

# Find Python
py = import('python').find_installation(pure: false)

# Cython files
pyx_files = files('werpy/metrics.pyx')

# Python Source files
py_files = files(
    'werpy/__init__.py',
    'werpy/errorhandler.py',
    'werpy/normalize.py',
    'werpy/summary.py',
    'werpy/summaryp.py',
    'werpy/wer.py',
    'werpy/werp.py',
    'werpy/werps.py',
    'werpy/wers.py'
)

# Install Python sources
py.install_sources(
    py_files,
    subdir: 'werpy',
    pure: false,
)

# Install py.typed file
install_data(
    'werpy/py.typed',
    install_dir: py.get_install_dir() / 'werpy'
)

# Find NumPy
numpy_dep = dependency('numpy')

# Compiler arguments for the Cython module.
# NPY_TARGET_VERSION pins the NumPy C API level the extension is built against;
# NPY_1_25_API_VERSION is the level provided by NumPy 1.26.
ext_c_args = ['-DNPY_TARGET_VERSION=NPY_1_25_API_VERSION']
ext_cython_args = []
if get_option('linetrace')
    ext_cython_args += ['-X', 'linetrace=True']
    ext_c_args += ['-DCYTHON_TRACE=1', '-DCYTHON_TRACE_NOGIL=1', '-DCYTHON_USE_SYS_MONITORING=0']
endif

# Cython module
py.extension_module(
    'metrics',
    pyx_files,
    dependencies : [numpy_dep],
    cython_args : ext_cython_args,
    c_args : ext_c_args,
    install : true,
    subdir : 'werpy'
)
