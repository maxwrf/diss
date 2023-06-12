#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <iostream>

class GNMClass
{
private:
    // int myPrivateVariable;

    void myPrivateFunction()
    {
        std::cout << "This is a private function." << std::endl;
    }

public:
    double **A_init;
    double **D;
    double **params;
    double **b;
    int m;
    int model;
    int n_p_combs;

    // define the constructor
    GNMClass(double **A_init_,
             double **D_,
             double **params_,
             double **b_,
             int m_,
             int model_,
             int n_p_combs_)
    {
        A_init = A_init_;
        D = D_;
        params = params_;
        b = b_;
        m = m_;
        model = model_;
        n_p_combs = n_p_combs_;
    }

    void runModel()
    {

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n_p_combs; j++)
            {
                b[i][j] = 0;
            }
        }
    }
};

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
    // TODO: This is very bad and slow
    npy_intp b_dims[] = {[0] = m,
                         [1] = n_p_combs};
    PyArrayObject *b = (PyArrayObject *)PyArray_SimpleNew(2,
                                                          b_dims,
                                                          NPY_DOUBLE);
    double **b_data;
    PyArray_AsCArray((PyObject **)&b,
                     &b_data,
                     b_dims,
                     2,
                     PyArray_DescrFromType(NPY_DOUBLE));

    // call the C native code
    GNMClass obj(A_init_data,
                 D_data,
                 params_data,
                 b_data,
                 m,
                 model,
                 n_p_combs);
    obj.runModel();

    return PyArray_Return(b);
}

static PyObject *version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.1");
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