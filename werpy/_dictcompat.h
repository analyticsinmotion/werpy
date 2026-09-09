/* SPDX-FileCopyrightText: 2023 Analytics in Motion <https://www.analyticsinmotion.com> */
/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef WERPY_DICTCOMPAT_H
#define WERPY_DICTCOMPAT_H

#include <Python.h>

#ifdef Py_LIMITED_API

/* Returns a borrowed reference to the value stored under key in dict,
 * inserting default_value first when key is absent. Returns NULL with
 * an exception set on error. */
static inline PyObject *
werpy_dict_setdefault(PyObject *dict, PyObject *key, PyObject *default_value)
{
    PyObject *value = PyDict_GetItemWithError(dict, key);
    if (value != NULL) {
        return value;
    }
    if (PyErr_Occurred()) {
        return NULL;
    }
    if (PyDict_SetItem(dict, key, default_value) < 0) {
        return NULL;
    }
    return default_value;
}

#else

#define werpy_dict_setdefault PyDict_SetDefault

#endif

#endif
