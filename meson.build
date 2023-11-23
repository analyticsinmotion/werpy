project(
    'werpy', 
    'c',
    version : '2.1.0',
    default_options : ['warning_level=3'])

# Find Python
py = import('python').find_installation('python3')

# Find Cython
cython = find_program('cython')

# Cython files
pyx_files = files('werpy/metrics.pyx')

# Compile Cython files
c_files = custom_target('cythonize',
                        output : 'metrics.c',
                        input : pyx_files,
                        command : [cython, '-3', '--fast-fail', '-o', '@OUTPUT@', '@INPUT@'])

# Source files
python_sources = files(
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
    python_sources,
    subdir: 'werpy',
    pure: false,
)


# Find NumPy include directories
numpy_includes = run_command(py, '-c', 'import numpy; print(numpy.get_include())').stdout().strip()


# Cython module
module1 = py.extension_module(
    'metrics',
    c_files,
    include_directories : include_directories(numpy_includes),
    install : true,
    subdir : 'werpy'
)
