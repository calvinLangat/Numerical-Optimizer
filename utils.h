#pragma once

#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>


typedef struct{
	char* chunk;
	char* idx;
	size_t size;
	size_t used;
}ARENA;


// Square matrices only

double CalcNorm2(double* vals, int numVars);

double Dot(const double* a, const double* b, size_t n);

void SetArrayZero(double* arr, size_t n);

void MatMul(double* A, double* B, double* C, size_t n);

void MatScalarMult(double* A, double scalar, size_t n);

void MatAdd(double* A, double* B, double* C, size_t n);

void MatSub(double* A, double* B, double* C, size_t n);

void MatSubSame(double* A, double* B, size_t n);

void MatVecMul(double* A, const double* b, double* c, size_t n);

void MatVecMulTranspose(double* A, const double* b, double* c, size_t n);

void MatVecMulTransposeRect(const double* A, const double* b, double* c, size_t rows, size_t cols);

void VecSub(double* a, double* b, double* c, size_t n);

void VecAdd(double* a, double* b, size_t n);

void VecAddMul(const double* a,const double* b, double* c, double scalar, size_t n);

void VecScalarMult(double* a, double scalar, size_t n);

void VecScalarMult2(double* a, double* b, double scalar, size_t n);

void OuterProduct(const double* a, const double* b, size_t n, double* Mat);

void CreateIdentityMat(double* A, size_t rows);

void CreateDiagonalMat(double* A, double* b, size_t rows);


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
int Householder_solve(double *A, const double *b, double *c, int n);

