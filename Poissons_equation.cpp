//cl /std:c++17 /openmp /O2 /EHsc Source.cpp /Fe:poisson.exe
//poisson.exe 128 10000 1e-8 8

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#define M_PI 3.14159265358979323846  

using namespace std; 

int idx(int i, int j, int N) 
{ 
    return i * N + j; 
} 

double resid(const vector<double>& u, const vector<double>& f, int N, double h2)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 1; i < N - 1; ++i)
    {
        for (int j = 1; j < N - 1; ++j) 
        {
            double Au = (-4.0 * u[idx(i, j, N)]
                + u[idx(i - 1, j, N)] + u[idx(i + 1, j, N)]
                + u[idx(i, j - 1, N)] + u[idx(i, j + 1, N)]) / h2; 
            double r = f[idx(i, j, N)] + Au;                              

            sum += r * r;
        }
    }
    return sqrt(sum);
}

int main(int argc, char** argv) //argc - число аргументов командной строки, argv - значения аргументов командной строки.
{
    if (argc < 4) 
    {
        fprintf(stderr, "Usage: %s N max_iter tol [num_threads]\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    const int max_iter = atoi(argv[2]);
    const double tol = atof(argv[3]);
    int threads = 0;

    if (N < 3) 
    { 
        fprintf(stderr, "N must be >= 3\n"); 
        return 1; 
    }
    if (max_iter <= 0) 
    { 
        fprintf(stderr, "max_iter must be > 0\n"); 
        return 1; 
    }
    if (threads > 0) 
    { 
        omp_set_num_threads(threads); 
    }

    const double h = 1.0 / (N - 1);
    const double h2 = h * h;

    //double omega = 2.0 / (1.0 + sin(M_PI * h));  
    double omega = 1;

    vector<double> u(N * N, 0.0);
    vector<double> f(N * N, 0.0);
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) 
        {
            f[idx(i, j, N)] = 1.0;
        }
    }

    double res0 = resid(u, f, N, h2);
    if (res0 == 0.0)
        res0 = 1.0;
    double res = res0;

    for (int it = 1; it <= max_iter; ++it) 
    {
#pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; ++i) 
        {
            for (int j = 1; j < N - 1; ++j) 
            {
                if (((i + j) & 1) == 0) { 
                    /*u[idx(i, j, N)] = 0.25 * (
                        u[idx(i - 1, j, N)] + u[idx(i + 1, j, N)] +
                        u[idx(i, j - 1, N)] + u[idx(i, j + 1, N)] -
                        h2 * f[idx(i, j, N)]
                        );*/

                    double unew = 0.25 * (u[idx(i - 1, j, N)] + u[idx(i + 1, j, N)]
                        + u[idx(i, j - 1, N)] + u[idx(i, j + 1, N)]
                        + h2 * f[idx(i, j, N)]);
                    u[idx(i, j, N)] = (1.0 - omega) * u[idx(i, j, N)] + omega * unew;
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; ++i) 
        {
            for (int j = 1; j < N - 1; ++j) 
            {
                if (((i + j) & 1) == 1) 
                {
                    /*u[idx(i, j, N)] = 0.25 * (
                        u[idx(i - 1, j, N)] + u[idx(i + 1, j, N)] +
                        u[idx(i, j - 1, N)] + u[idx(i, j + 1, N)] -
                        h2 * f[idx(i, j, N)]
                        );*/

                    double unew = 0.25 * (u[idx(i - 1, j, N)] + u[idx(i + 1, j, N)]
                        + u[idx(i, j - 1, N)] + u[idx(i, j + 1, N)]
                        + h2 * f[idx(i, j, N)]);
                    u[idx(i, j, N)] = (1.0 - omega) * u[idx(i, j, N)] + omega * unew;
                }
            }
        }

        if (it % 10 == 0 || it == max_iter) 
        {
            res = resid(u, f, N, h2);
           // printf("iter = %d, ||r||2 = %.6e (rel=%.6e)\n", it, res, res / res0);
            if (res / res0 < tol) {
                printf("Converged at iter = %d, ||r||2 = %.6e (rel=%.6e)\n",
                    it, res, res / res0);
                break;
            }
        }
        if (it == max_iter) 
        {
            res = resid(u, f, N, h2);
            printf("Stopped at iter = %d, ||r||2 = %.6e (rel=%.6e)\n", it, res, res / res0);
        }
    }
    double t1 = omp_get_wtime();
    printf("u(N/2,N/2) = %.8f\n", u[idx(N/2, N/2, N)]);
    printf("Elapsed time = %.6f seconds\n", t1 - t0);
    return 0;
}
