#include <stdio.h>
#include <Python.h>

int CfindPrimes(int num1, int num2)
{
    int flag_var, i, j;

    printf("Enter two range(input integer numbers only):");
    // Display prime numbers for input range
    printf("Prime numbers from %d and %d are:\n", num1, num2);
    for (i = num1 + 1; i < num2; ++i)
    {
        flag_var = 0;
        for (j = 2; j <= i / 2; ++j)
        {
            if (i % j == 0)
            {
                flag_var = 1;
                break;
            }
        }
        if (flag_var == 0)
            printf("%d\n", i);
    }
    return 0;
}

// int main()
// {
//     int a = 25;
//     int b = 3;
//     CfindPrimes(b, a);
//     return 0;
// }

static PyObject *findPrimes(PyObject *self, PyObject *args)
{
    int num1, num2, sts;
    if (!PyArg_ParseTuple(args, "ii", &num1, &num2))
        return NULL;
    sts = CfindPrimes(num1, num2);
    return PyLong_FromLong(sts);
}

static PyObject *version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.01");
}

static PyMethodDef Examples[] = {
    {"findPrimes", findPrimes, METH_VARARGS, "Calculating and print prime numbers between lower bound and upper bound "},
    {"version", (PyCFunction)version, METH_NOARGS, "return the version of the module"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef Example = {
    PyModuleDef_HEAD_INIT,
    "Example",
    "findPrimes Module",
    -1, // global state
    Examples};

// Initializer function
PyMODINIT_FUNC PyInit_Example(void)
{
    return PyModule_Create(&Example);
};