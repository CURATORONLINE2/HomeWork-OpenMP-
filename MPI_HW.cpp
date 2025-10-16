// cd C:\parallel_prog\MPI_HW\MPI_HW
// КОМПИЛЯЦИЯ: cl /O2 /std:c++17 /I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" MPI_HW.cpp /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib
// ЗАПУСК (MS-MPI): "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 2 .\MPI_HW.exe 1024 1000 1e-6

#include <mpi.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

using namespace std;

// глобальный индекс
inline size_t idx(int i, int j, int N) {
    return (size_t)i * (size_t)N + (size_t)j;
}

// разбиение строк между процессами
void make_dist(int N, int rank, int size, int& nloc, int& i0) {
    int q = N / size;
    int r = N % size;
    if (rank < r) 
        nloc = q + 1;
    else          
        nloc = q;
    i0 = rank * q + min(rank, r);
}

// Построение counts и displs (для MPI_Allgatherv)
void build_counts_displs(int N, int P, vector<int>& counts, vector<int>& displs) {
    counts.resize(P);
    displs.resize(P);
    int q = N / P, r = N % P;
    int off = 0;
    for (int p = 0; p < P; ++p) 
    {
        if (p < r)
            counts[p] = q + 1;
        else
            counts[p] = q;

        displs[p] = off;
        off += counts[p];
    }
}

// Генерация блока матрицы А, который будет считаться на каждом MPI-процессе
void fill_matrix(int nloc, int i0, int N, double dmin, double dmax, unsigned long long seed,
    vector<double>& A_local, vector<double>& diag_local)
{
    A_local.assign(nloc * (size_t)N, 0.0);
    diag_local.assign(nloc, 0.0);

    for (int i_loc = 0; i_loc < nloc; ++i_loc) {
        int i = i0 + i_loc;
        mt19937_64 gen(seed + i);

        uniform_real_distribution<double> dist_d(dmin, dmax);
        double aii = dist_d(gen);
        diag_local[i_loc] = aii;
        A_local[idx(i_loc, i, N)] = aii;

        uniform_real_distribution<double> dist_k(1.0, aii);
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            double k = dist_k(gen);
            A_local[idx(i_loc, j, N)] = k / (2.0 * aii * N);
        }
    }
}

void matvec_local(int nloc, int N,
    const vector<double>& A_local,
    const vector<double>& x_full,
    vector<double>& y_local)
{
    y_local.assign(nloc, 0.0);
    for (int i_loc = 0; i_loc < nloc; ++i_loc) {
        double s = 0.0;
        const double* row = &A_local[i_loc * (size_t)N];
        for (int j = 0; j < N; ++j) s += row[j] * x_full[j];
        y_local[i_loc] = s;
    }
}

// норма со сбором по всем процессам
double norm_all(const vector<double>& v, MPI_Comm comm)
{
    double local = 0.0;
    for (double x : v) local += x * x;
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return sqrt(global);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s N max_iter tol [seed] [dmin dmax]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const int max_iter = atoi(argv[2]);
    const double tol = atof(argv[3]);

    unsigned long long seed;
    if (argc >= 5) 
        seed = strtoull(argv[4], nullptr, 10);
    else           
        seed = 12345ull;

    double dmin, dmax;
    if (argc >= 7) {
        dmin = atof(argv[5]); 
        dmax = atof(argv[6]); 
    }
    else 
    { 
        dmin = 10.0;          
        dmax = 20.0; 
    }

    int nloc = 0, i0 = 0;
    make_dist(N, rank, size, nloc, i0);

    if (rank == 0) {
        printf("N=%d  np=%d  max_iter=%d  tol=%.3e  seed=%llu  diag~U[%.2f,%.2f]\n",
            N, size, max_iter, tol, (unsigned long long)seed, dmin, dmax);
    }

    vector<int> counts, displs;
    build_counts_displs(N, size, counts, displs);

    vector<double> A_local, diag_local;
    fill_matrix(nloc, i0, N, dmin, dmax, seed, A_local, diag_local);

    vector<double> x_true(N);
    {
        mt19937_64 gen(seed + 777);
        uniform_real_distribution<double> dist_x(-1.0, 1.0);
        for (int i = 0; i < N; ++i) x_true[i] = dist_x(gen);
    }

    vector<double> b_local;
    matvec_local(nloc, N, A_local, x_true, b_local);

    // Нормы
    double bnorm = norm_all(b_local, comm);
    if (bnorm == 0.0) bnorm = 1.0;

    double xtrue_norm = 0.0;
    {
        double loc = 0.0;
        for (int k = 0; k < nloc; ++k) {
            double v = x_true[i0 + k];
            loc += v * v;
        }
        MPI_Allreduce(&loc, &xtrue_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
        xtrue_norm = sqrt(xtrue_norm);
        if (xtrue_norm == 0.0) xtrue_norm = 1.0;
    }

    // Метод Якоби
    vector<double> x_old(N, 0.0);         
    vector<double> x_new_local(nloc, 0.0); 
    vector<double> x_new_full(N, 0.0);     
    vector<double> y_local, diff_local(nloc);

    MPI_Barrier(comm);
    double t_start = MPI_Wtime();

    double relres = 1.0, relerr = 1.0;
    int final_iter = max_iter;

    for (int it = 0; it < max_iter; ++it)
    {
        for (int i_loc = 0; i_loc < nloc; ++i_loc) {
            int i_gl = i0 + i_loc;
            const double* row = &A_local[i_loc * (size_t)N];

            double sum = 0.0;
            for (int j = 0; j < N; ++j) sum += row[j] * x_old[j];

            sum -= row[i_gl] * x_old[i_gl];

            x_new_local[i_loc] = (b_local[i_loc] - sum) / row[i_gl];
        }

        MPI_Allgatherv(x_new_local.data(), nloc, MPI_DOUBLE,
            x_new_full.data(), counts.data(), displs.data(),
            MPI_DOUBLE, comm);

        matvec_local(nloc, N, A_local, x_new_full, y_local);
        for (int k = 0; k < nloc; ++k) y_local[k] -= b_local[k];
        double res = norm_all(y_local, comm);
        relres = res / bnorm;

        // ошибка 
        for (int k = 0; k < nloc; ++k)
            diff_local[k] = x_new_full[i0 + k] - x_true[i0 + k];
        double err = norm_all(diff_local, comm);
        relerr = err / xtrue_norm;

        if (rank == 0 && (it % 10 == 0 || relres <= tol || it == max_iter - 1))
            printf("it=%5d  relres=%.3e  relerr=%.3e\n", it, relres, relerr);

        if (relres <= tol) { final_iter = it; break; }

        x_old.swap(x_new_full);
    }

    MPI_Barrier(comm);
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;

    if (rank == 0) {
        if (relres <= tol)
            printf("\nConverged at iter = %d\n", final_iter);
        else
            printf("\nStopped at iter = %d\n", max_iter);

        printf("Final relres=%.3e  relerr=%.3e\n", relres, relerr);
        printf("Wall time (solve): %.6f s\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
