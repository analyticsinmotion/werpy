project(
    'werpy',
    'c', 'cython',
    version : '3.5.0',
    license : 'BSD-3-Clause',
    meson_version : '>= 1.3.0',
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
pyx_files = files('werpy/_metrics.pyx')

# Python Source files
py_files = files(
    'werpy/__init__.py',
    'werpy/_errorhandler.py',
    'werpy/_metrics.pyi',
    'werpy/_normalize.py',
    'werpy/_summary.py',
    'werpy/_summaryp.py',
    'werpy/_wer.py',
    'werpy/_werp.py',
    'werpy/_werps.py',
    'werpy/_wers.py'
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

# Compiler arguments for the Cython module.
ext_c_args = []
ext_cython_args = []
if get_option('linetrace')
    ext_cython_args += ['-X', 'linetrace=True']
    ext_c_args += ['-DCYTHON_TRACE=1', '-DCYTHON_TRACE_NOGIL=1', '-DCYTHON_USE_SYS_MONITORING=0']
endif

# Cython module
py.extension_module(
    '_metrics',
    pyx_files,
    include_directories : include_directories('werpy'),
    cython_args : ext_cython_args,
    c_args : ext_c_args,
    limited_api : '3.11',
    install : true,
    subdir : 'werpy'
)
