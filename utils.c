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

int CreateArena(ARENA** arena, size_t size)
{
	ARENA* ar = *arena;
	ar->chunk = malloc(size);
	if (ar->chunk)
	{
		ar->size = size;
		ar->idx = ar->chunk;
		ar->used = 0;
		return 0;
	}
	return -1;	//failed malloc
}

void DestroyArena(ARENA* arena)
{
	free(arena->chunk);
	arena->chunk = NULL;
	arena->idx   = NULL;
	arena->size  = 0;
	arena->used  = 0;
}

char* ArenaAlloc(ARENA* arena, size_t size)
{
	if ((arena->size - arena->used) >= size)
	{
		char* pos = NULL;
		pos = arena->idx;
		arena->used += size;
		arena->idx  += size;
		return pos;
	}
	else
	{
		return NULL;
	}
}

