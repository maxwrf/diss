#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <iostream>

double Cnative(double **A_init,
               double **D,
               double **params,
               double *b,
               int m,
               int model)
{

    return 1;
}

static PyObject *hello(PyObject *self, PyObject *args)
{

    PyArrayObject *A_init, *D, *params;
    int m, model;

    if (!PyArg_ParseTuple(args, "OOOii", &A_init, &D, &params, &m, &model))
    {
        return NULL;
    }

    // Check that args are numpy arrays
    if (!PyArray_Check(A_init) || !PyArray_Check(D) || !PyArray_Check(params))
    {
        PyErr_SetString(PyExc_TypeError,
                        "spike trains and time need to be np arrays of doubles");
    };

    // prepare A_init and D as C objects
    int n_nodes = sqrt(PyArray_SIZE(A_init));
    int n_p_combs = PyArray_SIZE(params) / 2;
    double **A_init_data, **D_data, **params_data;

    npy_intp A_init_dims[] = {[0] = n_nodes, [1] = n_nodes};
    npy_intp D_dims[] = {[0] = n_nodes, [1] = n_nodes};
    npy_intp params_dims[] = {[0] = n_p_combs, [1] = 2};

    PyArray_AsCArray((PyObject **)&A_init,
                     &A_init_data,
                     A_init_dims,
                     2,
                     PyArray_DescrFromType(NPY_DOUBLE));

    PyArray_AsCArray((PyObject **)&D,
                     &D_data,
                     D_dims,
                     2,
                     PyArray_DescrFromType(NPY_DOUBLE));

    PyArray_AsCArray((PyObject **)&params,
                     &params_data,
                     params_dims,
                     2,
                     PyArray_DescrFromType(NPY_DOUBLE));

    // construct b the results matrix
    npy_intp b_dims[] = {[0] = m,
                         [1] = n_p_combs};
    PyArrayObject *b = (PyArrayObject *)PyArray_SimpleNew(2, b_dims, NPY_INT);
    double *b_data = (double *)PyArray_DATA(b);

    // call the C native code
    double res = Cnative(A_init_data,
                         D_data,
                         params_data,
                         b_data,
                         m,
                         model);

    double element = A_init_data[0][0];

    printf("%f\n", element);

    return PyFloat_FromDouble(res);
}

static PyObject *version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.01");
}

static PyMethodDef methods[] = {
    {"hello", hello, METH_VARARGS, "Desc"},
    {"version", (PyCFunction)version, METH_NOARGS, "Desc"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef GNMC = {
    PyModuleDef_HEAD_INIT,
    "GNMC",
    "gnm Module",
    -1,
    methods};

// Initializer
PyMODINIT_FUNC PyInit_GNMC(void)
{
    PyObject *mod = PyModule_Create(&GNMC);
    import_array();
    return mod;
};