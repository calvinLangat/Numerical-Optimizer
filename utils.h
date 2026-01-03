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

void VecSub(double* a, double* b, double* c, size_t n);

void VecAdd(double* a, double* b, size_t n);

void VecAddMul(const double* a,const double* b, double* c, double scalar, size_t n);

void VecScalarMult(double* a, double scalar, size_t n);

void VecScalarMult2(double* a, double* b, double scalar, size_t n);

void OuterProduct(const double* a, const double* b, size_t n, double* Mat);

void CreateIdentityMat(double* A, size_t rows);

int CreateArena(ARENA** arena, size_t size);

void DestroyArena(ARENA* arena);

char* ArenaAlloc(ARENA* arena, size_t size);