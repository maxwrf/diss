#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

double run_P(int na,
             int nb,
             double dt,
             double *sta_data,
             double *stb_data)
{
    /* Calculate the term P_1. the fraction of spikes from train 1 that
     * are within +/- dt of train 2.
     */

    int i, j, N12;

    N12 = 0;
    j = 0;
    for (i = 0; i <= (na - 1); i++)
    {
        while (j < nb)
        {
            // check every spike in train 1 to see if there's a spike in
            //  train 2 within dt  (don't count spike pairs)
            //  don't need to search all j each iteration
            if (fabs(sta_data[i] - stb_data[j]) <= dt)
            {
                N12 = N12 + 1;
                break;
            }
            else if (stb_data[j] > sta_data[i])
            {
                break;
            }
            else
            {
                j = j + 1;
            }
        }
    }
    return N12;
}

double run_T(int n,
             double dt,
             double start,
             double end,
             double *spike_times_1)
{

    /* Calculate T_A, the fraction of time 'tiled' by spikes with +/- dt.
     *
     * This calculation requires checks to see that (a) you don't count
     * time more than once (when two or more tiles overlap) and checking
     * beg/end of recording.
     */

    double time_A;
    int i = 0;
    double diff;

    // maximum
    time_A = 2 * (double)n * dt;

    // Assume at least one spike in train!

    // if just one spike in train
    if (n == 1)
    {

        if ((spike_times_1[0] - start) < dt)
        {
            time_A = time_A - start + spike_times_1[0] - dt;
        }
        else if ((spike_times_1[0] + dt) > end)
        {
            time_A = time_A - spike_times_1[0] - dt + end;
        }
    }

    else
    { /* more than one spike in train */
        while (i < (n - 1))
        {
            diff = spike_times_1[i + 1] - spike_times_1[i];
            if (diff < 2 * dt)
            {
                // subtract overlap
                time_A = time_A - 2 * dt + diff;
            }

            i++;
        }

        // check if spikes are within dt of the start and/or end, if so
        // just need to subract overlap of first and/or last spike as all
        // within-train overlaps have been accounted for (in the case that
        // more than one spike is within dt of the start/end

        if ((spike_times_1[0] - start) < dt)
        {
            time_A = time_A - start + spike_times_1[0] - dt;
        }
        if ((end - spike_times_1[n - 1]) < dt)
        {
            time_A = time_A - spike_times_1[n - 1] - dt + end;
        }
    }
    return time_A;
}

double Csttc(double *st1_data,
             double *st2_data,
             int n1,
             int n2,
             double dt,
             double *time_data)
{
    double TA, TB, PA, PB, T;

    T = time_data[1] - time_data[0];

    TA = run_T(n1, dt, time_data[0], time_data[1], st1_data);
    TA = TA / T;

    TB = run_T(n2, dt, time_data[0], time_data[1], st2_data);
    TB = TB / T;

    PA = run_P(n1, n2, dt, st1_data, st2_data);
    PA = PA / (double)n1;

    PB = run_P(n2, n1, dt, st2_data, st1_data);
    PB = PB / (double)n2;

    return (0.5 * (PA - TB) / (1 - TB * PA) + 0.5 * (PB - TA) / (1 - TA * PB));
}

static PyObject *sttc(PyObject *self, PyObject *args)
{
    double res;

    // read in numpy array
    PyArrayObject *st1, *st2, *time;
    double dt;

    if (!PyArg_ParseTuple(args, "OOdO", &st1, &st2, &dt, &time))
        return NULL;

    // Check that arg is numpy array
    if (!PyArray_Check(st1) || !PyArray_Check(st1) || !PyArray_Check(time))
    {
        PyErr_SetString(PyExc_TypeError, "spike trains and time need to be np arrays of doubles");
    };

    // cast
    int n1 = PyArray_SIZE(st1);
    int n2 = PyArray_SIZE(st2);
    int n_time = PyArray_SIZE(time);

    if (n_time != 2)
    {
        PyErr_SetString(PyExc_TypeError, "time needs to have length 2");
    }

    double *st1_data, *st2_data, *time_data;
    npy_intp st1_dims[] = {[0] = n1};
    npy_intp st2_dims[] = {[0] = n2};
    npy_intp times_dims[] = {[0] = 2};

    PyArray_AsCArray((PyObject **)&st1, &st1_data, st1_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
    PyArray_AsCArray((PyObject **)&st2, &st2_data, st2_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
    PyArray_AsCArray((PyObject **)&time, &time_data, times_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));
    // double *data = (double *)PyArray_DATA(arr);

    // call the C native code
    res = Csttc(st1_data, st2_data, n1, n2, dt, time_data);

    // return PyLong_FromLong(res);
    return PyFloat_FromDouble(res);
}

static PyObject *tiling(PyObject *self, PyObject *args)
{
    int i, n_arrays;
    PyObject *list;

    if (!PyArg_ParseTuple(args, "O", &list))
    {
        return NULL;
    };

    n_arrays = PyObject_Length(list);
    printf("number of arrays: %d", n_arrays);
    for (i = 0; i < n_arrays; i++)
    {
        PyArrayObject *elem;
        elem = (PyArrayObject *)PyList_GetItem(list, i);

        int n1 = PyArray_SIZE(elem);
        double *np_array;
        npy_intp st1_dims[] = {[0] = n1};
        PyArray_AsCArray((PyObject **)&elem, &np_array, st1_dims, 1, PyArray_DescrFromType(NPY_DOUBLE));

        // np_array = (double *)PyArray_DATA(elem);
        printf("Value 0 : %.10f\n", *np_array);
    }

    return PyFloat_FromDouble(0);
}

static PyObject *version(PyObject *self)
{
    return Py_BuildValue("s", "Version 0.01");
}

static PyMethodDef methods[] = {
    {"sttc", sttc, METH_VARARGS, "Calculating and print prime numbers between lower bound and upper bound "},
    {"tiling", tiling, METH_VARARGS, "Computes matrix of sttc's"},
    {"version", (PyCFunction)version, METH_NOARGS, "return the version of the module"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef STTC = {
    PyModuleDef_HEAD_INIT,
    "STTC",
    "sttc Module",
    -1, // global state
    methods};

// Initializer
PyMODINIT_FUNC PyInit_STTC(void)
{
    PyObject *mod = PyModule_Create(&STTC);
    import_array();
    return mod;
};