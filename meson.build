project(
    'werpy_test_v4', 
    'c',
    version : '0.0.1',
    default_options : ['warning_level=3'])

py = import('python').find_installation(pure: false)

cython = find_program('cython')
pyx_files = files('werpy_test_v4/metrics.pyx')
c_files = custom_target('cythonize',
                        output : 'metrics.c',
                        input : pyx_files,
                        command : [cython, '-3', '--fast-fail', '-o', '@OUTPUT@', '@INPUT@'])

py.extension_module(
    'werpy_test_v4',
    c_files,
    install : true
)
