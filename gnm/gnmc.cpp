#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <iostream>
#include <vector>

// GNM rules
// 0: 'spatial',
// 1: 'neighbors',
// 2: 'matching',
// 3: 'clu-avg',
// 4: 'clu-min',
// 5: 'clu-max',
// 6: 'clu-dist',
// 7: 'clu-prod',
// 8: 'deg-avg',
// 9: 'deg-min',
// 10: 'deg-max',
// 11: 'deg-dist',
// 12: 'deg-prod'

class GNMClass
{
private:
    double **A_current, **K_current;
    int *k_current; // stores the nodal degree
    double epsilon = 1e-5;
    double *clu_coeff_current;

    void compute_clustering_coeff()
    {
        // This implementation is only correct for undirected graphs

        clu_coeff_current = new double[n_nodes];
        for (int i_node = 0; i_node < n_nodes; ++i_node)
        {
            if (k_current[i_node] > 1) // only if there are more than one neighbors
            {
                // get the neighbors
                int neighbors[k_current[i_node]];
                int n_neighbors = 0;
                for (int j = 0; j < n_nodes; ++j)
                {
                    if (A_current[i_node][j])
                    {
                        neighbors[n_neighbors] = j;
                        n_neighbors++;
                    }
                }

                // get the connections across neighbors
                int sum_S = 0;
                for (int i = 0; i < n_neighbors; ++i)
                {
                    for (int j = i + 1; j < n_neighbors; ++j)
                    {
                        sum_S += A_current[neighbors[i]][neighbors[j]];
                    }
                }

                clu_coeff_current[i_node] = (double)(2 * sum_S) / (double)(k_current[i_node] * (k_current[i_node] - 1));
            }
            else
            {
                clu_coeff_current[i_node] = 0;
            }
        }
    }

    std::vector<int> update_clustering_coeff(int uu, int vv)
    {
        // get the neighbors at row node (uu)
        int uu_neighbors[k_current[uu]];
        int uu_n_neighbors = 0;

        // get the neighbors at col node (vv)
        int vv_neighbors[k_current[vv]];
        int vv_n_neighbors = 0;

        // get common neighbor at row and col nodes
        int uv_n_neighbors = 0;
        std::vector<int> uv_neighbors;

        for (int i = 0; i < n_nodes; ++i)
        {
            // row neighbor
            if (A_current[uu][i])
            {
                uu_neighbors[uu_n_neighbors] = i;
                uu_n_neighbors++;
            }

            // col neighbor
            if (A_current[vv][i])
            {
                vv_neighbors[vv_n_neighbors] = i;
                vv_n_neighbors++;
            }

            // common neighbors
            if ((A_current[uu][i]) && (A_current[vv][i]))
            {
                uv_n_neighbors++;
                uv_neighbors.push_back(i);
            }
        }

        // get connections across row neighbors and update
        int sum_S_uu = 0;
        for (int i = 0; i < uu_n_neighbors; ++i)
        {
            for (int j = i + 1; j < uu_n_neighbors; ++j)
            {
                sum_S_uu += A_current[uu_neighbors[i]][uu_neighbors[j]];
            }
        }
        clu_coeff_current[uu] = (double)(2 * sum_S_uu) / (double)(k_current[uu] * (k_current[uu] - 1));

        // get connections across col neighbors and update
        int sum_S_vv = 0;
        for (int i = 0; i < vv_n_neighbors; ++i)
        {
            for (int j = i + 1; j < vv_n_neighbors; ++j)
            {
                sum_S_vv += A_current[vv_neighbors[i]][vv_neighbors[j]];
            }
        }
        clu_coeff_current[vv] = (double)(2 * sum_S_vv) / (double)(k_current[vv] * (k_current[vv] - 1));

        // get connections across common neighbors and update (always + 2/possible)
        for (int i = 0; i < uv_neighbors.size(); i++)
        {
            clu_coeff_current[i] = clu_coeff_current[i] + (double)2 / (double)(k_current[i] * (k_current[i] - 1));
        }

        // cleanup
        for (int i = 0; i < n_nodes; i++)
        {
            if (k_current[i] < 3)
            {
                clu_coeff_current[i] = 0;
            }
        }

        return uv_neighbors;
    }

    void reset_A_current()
    {
        // Initialize a copy of A_init to work on
        A_current = new double *[n_nodes];
        for (int i = 0; i < n_nodes; ++i)
        {
            A_current[i] = new double[n_nodes];
            for (int j = 0; j < n_nodes; ++j)
            {
                A_current[i][j] = A_init[i][j];
            }
        }
    }

    void init_K()
    // Initialize the value matrix
    // TODO: You can probably speed this up by only iterating over the upper triangle
    {
        K_current = new double *[n_nodes];
        switch (model)
        {
        case 0: // spatial
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = 1.0;
                }
            }
            break;

        case 3: // clu-avg
            compute_clustering_coeff();
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = ((clu_coeff_current[i] + clu_coeff_current[j]) / 2) + epsilon;
                }
            }
            break;

        case 8: // deg-avg
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = ((k_current[i] + k_current[j]) / 2) + epsilon;
                }
            }
            break;

        case 9: // deg-min
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = ((k_current[i] > k_current[j]) ? k_current[j] : k_current[i]) + epsilon;
                }
            }
            break;

        case 10: // deg-max
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = ((k_current[i] > k_current[j]) ? k_current[i] : k_current[j]) + epsilon;
                }
            }
            break;

        case 11: // deg-dist
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = fabs(k_current[i] - k_current[j]) + epsilon;
                }
            }
            break;

        case 12: // deg-prod
            for (int i = 0; i < n_nodes; ++i)
            {
                K_current[i] = new double[n_nodes];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = k_current[i] * k_current[j] + epsilon;
                }
            }
            break;
        }
    }

    void update_K(std::vector<int> bth)
    {
        switch (model)
        {
        case 3: // clu-avg
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] + clu_coeff_current[j]) / 2) + epsilon;
                }
            }
            break;
        case 8: // deg-avg
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = K_current[j][i] = ((k_current[i] + k_current[j]) / 2) + epsilon;
                }
            }
            break;

        case 9: // deg-min
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = K_current[j][i] = ((k_current[i] > k_current[j]) ? k_current[j] : k_current[i]) + epsilon;
                }
            }
            break;

        case 10: // deg-max
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = K_current[j][i] = ((k_current[i] > k_current[j]) ? k_current[i] : k_current[j]) + epsilon;
                }
            }
            break;

        case 11: // deg-dist
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = K_current[j][i] = fabs(k_current[i] - k_current[j]) + epsilon;
                }
            }
            break;

        case 12: // deg-prod
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    K_current[i][j] = K_current[j][i] = k_current[i] * k_current[j] + epsilon;
                }
            }
            break;

        default: // spatial
            break;
        }
    }

    void run_param_comb(int i_pcomb)
    // main function for the generative network build
    {
        // get the params
        double eta = params[i_pcomb][0];
        double gamma = params[i_pcomb][1];

        // initiate the degree of each node
        k_current = new int[n_nodes];
        for (int i = 0; i < n_nodes; ++i)
        {
            k_current[i] = 0;
            for (int j = 0; j < n_nodes; ++j)
            {
                k_current[i] += A_current[i][j];
            }
        }
        // compute the inital value matrix
        init_K();

        // initiate cost and value matrices
        double **Fd = new double *[n_nodes];
        double **Fk = new double *[n_nodes];
        double **Ff = new double *[n_nodes];

        for (int i = 0; i < n_nodes; ++i)
        {
            Fd[i] = new double[n_nodes];
            Fk[i] = new double[n_nodes];
            Ff[i] = new double[n_nodes];

            for (int j = 0; j < n_nodes; ++j)
            {
                Fd[i][j] = pow(D[i][j], eta);
                Fk[i][j] = pow(K_current[i][j], gamma);
                Ff[i][j] = Fd[i][j] * Fk[i][j] * (A_current[i][j] == 0);
            }
        }

        // get the indices of the upper triangle of P (denoted u and v)
        int *u = new int[n_nodes * (n_nodes - 1) / 2];
        int *v = new int[n_nodes * (n_nodes - 1) / 2];
        int upper_tri_index = 0;

        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i + 1; j < n_nodes; ++j)
            {
                u[upper_tri_index] = i;
                v[upper_tri_index] = j;
                upper_tri_index++;
            }
        }

        // initiate P
        double *P = new double[upper_tri_index];
        for (int i = 0; i < upper_tri_index; ++i)
        {
            P[i] = Ff[u[i]][v[i]];
        }

        // Number of connections we start with
        int m_seed = 0;
        for (int i = 0; i < upper_tri_index; ++i)
        {
            if (A_current[u[i]][v[i]] != 0)
            {
                m_seed++;
            }
        }

        // main loop adding new connections to adjacency matrix
        for (int i = m_seed + 1; i <= m; ++i)
        {
            // compute the cumulative sum of the probabilities
            double *C = new double[upper_tri_index + 1];
            C[0] = 0;
            for (int j = 0; j < upper_tri_index; ++j)
            {
                C[j + 1] = C[j] + P[j];
            }

            // select an element
            double r = std::rand() / (RAND_MAX + 1.0) * C[upper_tri_index];
            int selected = 0;
            while (selected < upper_tri_index && r >= C[selected + 1])
            {
                selected++;
            }
            int uu = u[selected];
            int vv = v[selected];

            // update the node degree array
            k_current[uu] += 1;
            k_current[vv] += 1;

            // update the adjacency matrix
            A_current[uu][vv] = A_current[vv][uu] = 1;

            // get the node indices for update
            std::vector<int> bth;
            if ((model > 2) && (model < 8))
            {
                bth = update_clustering_coeff(uu, vv);
            }
            bth.push_back(uu);
            bth.push_back(vv);

            // update K matrix
            update_K(bth);

            // update Ff matrix (probabilities)
            for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
            {
                int i = bth[bth_i];
                for (int j = 0; j < n_nodes; ++j)
                {
                    Ff[i][j] = Ff[j][i] = Fd[i][j] * pow(K_current[i][j], gamma) * (A_current[i][j] == 0);
                }
            }

            // update p
            // TODO: This currently just updates every position
            for (int i = 0; i < upper_tri_index; ++i)
            {
                P[i] = Ff[u[i]][v[i]];
            }
        }

        // update b with the result
        int nth_edge = 0;
        for (int i = 0; i < upper_tri_index; ++i)
        {
            if (A_current[u[i]][v[i]])
            {
                b[nth_edge][i_pcomb] = (u[i] + 1) * pow(10, ceil(log10(v[i] + 1))) + v[i];
                nth_edge = nth_edge + 1;
            }
        }
    }

public:
    double **A_init;
    double **D;
    double **params;
    int **b;
    int m;
    int model;
    int n_p_combs;
    int n_nodes;

    // define the constructor
    GNMClass(double **A_init_,
             double **D_,
             double **params_,
             int **b_,
             int m_,
             int model_,
             int n_p_combs_,
             int n_nodes_)
    {
        A_init = A_init_;
        D = D_;
        params = params_;
        b = b_;
        m = m_;
        model = model_;
        n_p_combs = n_p_combs_;
        n_nodes = n_nodes_;
    }

    void generateModels()
    // Initiates the network generation leveraging the different rules
    {
        for (int i_pcomb = 0; i_pcomb < n_p_combs; i_pcomb++)
        {
            reset_A_current();
            run_param_comb(i_pcomb);
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
                                                          NPY_INT);
    int **b_data;
    PyArray_AsCArray((PyObject **)&b,
                     &b_data,
                     b_dims,
                     2,
                     PyArray_DescrFromType(NPY_INT));

    // call the C native code
    GNMClass obj(A_init_data,
                 D_data,
                 params_data,
                 b_data,
                 m,
                 model,
                 n_p_combs,
                 n_nodes);

    obj.generateModels();

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