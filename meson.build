project(
    'werpy', 
    'c', 'cython',
    version : '3.1.1',
    license: 'BSD-3',
    meson_version: '>= 1.1.0',
    default_options : [
        'buildtype=debugoptimized',
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

# Find NumPy include directories
#numpy_includes = run_command(py, '-c', 'import numpy; print(numpy.get_include())', check: true).stdout().strip()
numpy_dep = dependency('numpy')

# Cython module
py.extension_module(
    'metrics',
    pyx_files,
    #include_directories : include_directories(numpy_includes),
    dependencies : [numpy_dep],
    install : true,
    subdir : 'werpy'
)
