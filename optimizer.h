#include "utils.h"

typedef double (*EVALUATE)(double*);	// Function pointer signature for functions to be evaluated

void ComputeGradient(EVALUATE eval, double* x0, int numVars, double* gradient);

void BfgsInverseUpdate(double* H, const double* s, const double* y, size_t n);

int Optimize_Steepest(EVALUATE eval, double* x0, int numVars, double stepLength, double* result);

int Optimize_QuasiNewton_FixedStep(EVALUATE eval, double* x0, int numVars, double* H0, double* result);

double CalculateStepLengthWeakWolfe(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars);

double CalculateStepLengthStrongWolfe(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars);

double CalculateStepLengthArmijo(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars);