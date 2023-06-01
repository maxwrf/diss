#include <Python.h>

int sttc(int num1, int num2)
{
    int i, j;

    i = num1;
    j = num2;

    return i;
}

static PyObject *findPrimes(PyObject *self, PyObject *args)
{
    int num1, num2, res;
    if (!PyArg_ParseTuple(args, "ii", &num1, &num2))
        return NULL;
    res = sttc(num1, num2);
    return PyLong_FromLong(res);
}

static PyObject *version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.01");
}

static PyMethodDef Examples[] = {
    {"findPrimes", findPrimes, METH_VARARGS, "Calculating and print prime numbers between lower bound and upper bound "},
    {"version", (PyCFunction)version, METH_NOARGS, "return the version of the module"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef STTC = {
    PyModuleDef_HEAD_INIT,
    "Example",
    "findPrimes Module",
    -1, // global state
    Examples};

// Initializer function
PyMODINIT_FUNC PyInit_Example(void)
{
    return PyModule_Create(&STTC);
};