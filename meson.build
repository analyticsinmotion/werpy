project(
    'werpy', 
    'c', 'cython',
    version : '3.0.1',
    license: 'BSD-3',
    meson_version: '>= 1.1.0',
    default_options : [
        'buildtype=debugoptimized',
        'warning_level=3',
        'c_std=c17'
        ]
)

# Find Python
py = import('python').find_installation('python3')

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

# Find NumPy include directories
numpy_includes = run_command(py, '-c', 'import numpy; print(numpy.get_include())').stdout().strip()

# Cython module
py.extension_module(
    'metrics',
    pyx_files,
    include_directories : include_directories(numpy_includes),
    install : true,
    subdir : 'werpy'
)
