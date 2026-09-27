#include "utils.h"

void SetArrayZero(double* arr, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		arr[i] = 0;
	}
}

double CalcNorm2(double* vals, int numVars)
{
	double sum_sqrs = 0;
	for (int i = 0; i < numVars; ++i)
	{
		sum_sqrs += pow(vals[i], 2);
	}
	return sqrt(sum_sqrs);
}

void MatMul(double* A, double* B, double* C, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			C[i * n + j] = 0;
			for (int k = 0; k < n; ++k)
			{
				C[i * n + j] += A[i * n + k] * B[k * n + j];
			}
		}
	}
}

void MatScalarMult(double* A, double scalar, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		A[i] *= scalar;
	}
}

void MatAdd(double* A, double* B, double* C, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		C[i] = A[i] + B[i];
	}
}

void MatSub(double* A, double* B, double* C, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		C[i] = A[i] - B[i];
	}
}

void MatSubSame(double* A, double* B, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		A[i] -= B[i];
	}
}

void MatVecMul(double* A, const double* b, double* c, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		c[i] = 0;
		for (int j = 0; j < n; ++j)
		{
			c[i] += A[i * n + j] * b[j];
		}
	}
}

void MatVecMulTranspose(double* A, const double* b, double* c, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		c[i] = 0;
		for (int j = 0; j < n; ++j)
		{
			c[i] += A[j * n + i] * b[j];
		}
	}
}

void MatVecMulTransposeRect(
    const double* A,
    const double* b,
    double* c,
    size_t rows,
    size_t cols)
{
    for (size_t i = 0; i < cols; ++i)
    {
        c[i] = 0.0;

        for (size_t j = 0; j < rows; ++j)
        {
            c[i] += A[j * cols + i] * b[j];
        }
    }
}

double Dot(const double* a, const double* b, size_t n)
{
	double sum = 0.0;
	for (int i = 0; i < n; ++i)
	{
		sum += a[i] * b[i];
	}

	return sum;
}

void VecSub(double* a, double* b, double* c, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		c[i] = a[i] - b[i];
	}
}

void VecAdd(double* a, double* b, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		a[i] += b[i];
	}
}

void VecAddMul(const double* a,const double* b, double* c, double scalar, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		c[i] = a[i] + b[i];
		c[i] *= scalar;
	}
}

void VecScalarMult(double* a, double scalar, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		a[i] *= scalar;
	}
}

void VecScalarMult2(double* a, double* b, double scalar, size_t n)
{
	for (int i = 0; i < n; ++i)
	{
		b[i] = a[i] * scalar;
	}
}


void OuterProduct(const double* a, const double* b, size_t n, double* Mat)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			Mat[i * n + j] = a[i] * b[j];
		}
	}
}

void CreateIdentityMat(double* A, size_t rows)
{
	int j = 0;
	for (int i = 0; i < rows; ++i)
	{
		A[i * rows + j] = 1;
		++j;
	}
}

void CreateDiagonalMat(double* A, double* b, size_t rows)
{
	for(int i=0; i<rows;++i)
	{
		A[i * rows + i] = b[i];
	}
}

/* A: row-major n-by-n matrix; overwritten with R.
 * b: right-hand side; unchanged.
 * c: output array of n doubles; receives x.
 *
 * A, b, and c must not overlap.
 *
 * Returns:
 *   0  success
 *  -1  invalid arguments
 *  -2  allocation failed
 *  -3  singular matrix
 */
int Householder_solve(double *A, const double *b, double *c, int n)
{
    if (A == NULL || b == NULL || c == NULL || n <= 0)
        return -1;

    size_t N = (size_t)n;
    double *v = malloc(N * sizeof(*v));
    if (v == NULL)
        return -2;

    for (int i = 0; i < n; ++i)
        c[i] = b[i];

    for (int k = 0; k < n - 1; ++k) {
        /* Construct the reflector vector. */
        double norm = 0.0;
        for (int i = k; i < n; ++i) {
            v[i] = A[(size_t)i * N + k];
            norm = hypot(norm, v[i]);
        }

        if (norm == 0.0) {
            free(v);
            return -3;
        }

        double alpha = (v[k] >= 0.0) ? -norm : norm;

        /* Scaling v does not change the reflection. */
        for (int i = k; i < n; ++i)
            v[i] /= norm;

        v[k] -= alpha / norm;

        double vv = 0.0;
        for (int i = k; i < n; ++i)
            vv += v[i] * v[i];

        double tau = 2.0 / vv;

        /* Transform the remaining columns. */
        for (int j = k + 1; j < n; ++j) {
            double dot = 0.0;
            for (int i = k; i < n; ++i)
                dot += v[i] * A[(size_t)i * N + j];

            double s = tau * dot;
            for (int i = k; i < n; ++i)
                A[(size_t)i * N + j] -= s * v[i];
        }

        /* Transform the working right-hand side in c. */
        double dot = 0.0;
        for (int i = k; i < n; ++i)
            dot += v[i] * c[i];

        double s = tau * dot;
        for (int i = k; i < n; ++i)
            c[i] -= s * v[i];

        A[(size_t)k * N + k] = alpha;
        for (int i = k + 1; i < n; ++i)
            A[(size_t)i * N + k] = 0.0;
    }

    free(v);

    /* Back substitution: c becomes the solution. */
    for (int i = n - 1; i >= 0; --i) {
        double diagonal = A[(size_t)i * N + i];

        if (diagonal == 0.0)
            return -3;

        double sum = 0.0;
        for (int j = i + 1; j < n; ++j)
            sum += A[(size_t)i * N + j] * c[j];

        c[i] = (c[i] - sum) / diagonal;
    }

    return 0;
}