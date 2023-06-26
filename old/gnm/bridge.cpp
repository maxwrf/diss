#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <vector>
#include "gnm.h" 


static PyObject *get_gnms(PyObject *self, PyObject *args)
{

    PyArrayObject *A_Y, *A_init, *D, *params;
    int m, model;

    if (!PyArg_ParseTuple(args, "OOOOii", &A_Y, &A_init, &D, &params, &m, &model))
    {
        return NULL;
    }

    // Check that args are numpy arrays
    if (!PyArray_Check(A_Y) || !PyArray_Check(A_init) || !PyArray_Check(D) || !PyArray_Check(params))
    {
        PyErr_SetString(PyExc_TypeError,
                        "spike trains and time need to be np arrays of doubles");
    };

    // prepare A, A_init and D as C objects
    int n_nodes = sqrt(PyArray_SIZE(A_init));
    int n_p_combs = PyArray_SIZE(params) / 2;
    double **A_Y_data, **A_init_data, **D_data, **params_data;

    npy_intp A_init_dims[] = {[0] = n_nodes, [1] = n_nodes};
    npy_intp D_dims[] = {[0] = n_nodes, [1] = n_nodes};
    npy_intp params_dims[] = {[0] = n_p_combs, [1] = 2};

    PyArray_AsCArray((PyObject **)&A_Y,
                     &A_Y_data,
                     A_init_dims,
                     2,
                     PyArray_DescrFromType(NPY_DOUBLE));

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
    // TODO: Is this the best way to do this?
    int **b_data;
    npy_intp b_dims[] = {[0] = m,
                         [1] = n_p_combs};
    PyArrayObject *b = (PyArrayObject *)PyArray_SimpleNew(2,
                                                          b_dims,
                                                          NPY_INT);

    PyArray_AsCArray((PyObject **)&b,
                     &b_data,
                     b_dims,
                     2,
                     PyArray_DescrFromType(NPY_INT));

    // construct K results matrix
    double **K_data;
    npy_intp K_dims[] = {[0] = 4,
                         [1] = n_p_combs};
    PyArrayObject *K = (PyArrayObject *)PyArray_SimpleNew(2,
                                                          K_dims,
                                                          NPY_DOUBLE);

    PyArray_AsCArray((PyObject **)&K,
                     &K_data,
                     K_dims,
                     2,
                     PyArray_DescrFromType(NPY_DOUBLE));

    // call the C native code
    GNM obj(A_Y_data,
            A_init_data,
            D_data,
            params_data,
            b_data,
            K_data,
            m,
            model,
            n_p_combs,
            n_nodes);

    obj.generateModels();

    // prepare tuple to return
    PyObject *result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, PyArray_Return(b));
    PyTuple_SetItem(result, 1, PyArray_Return(K));

    return result;
}

static PyObject *version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.1");
}

static PyMethodDef methods[] = {
    {"get_gnms", get_gnms, METH_VARARGS, "Desc"},
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