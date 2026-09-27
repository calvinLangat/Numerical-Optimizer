#include "utils.h"

typedef double (*EVALUATE)(double*);	// Function pointer signature for functions to be evaluated

typedef struct {
	EVALUATE func;					// Objective function
	EVALUATE* equalityConstFunc;	// Pointer to array of equality constraints function pointers
	EVALUATE* inEqualityConstFunc;	// Pointer to array of inequality constraints function pointers

	size_t numVarsFunc;				// Number of variables for the functions
	size_t numEqualityConst;		// Number of Equality constraints
	size_t numInEqualityConst;		// Number of Inequality constraints

	double* x0;						// Pointer to current x;
	double* lambdas;				// Pointer to Lagrange multipliers for equality constraints
	double* sigmas;					// Pointer to Lagrange multipliers for inequality constraints
	double* slacks;					// Pointer to slack variables for inequality constraints
	double  mu;						// Penalty variable for log barrier slack variables

	double* gradWRTvars;			// Gradient of lagrangian w.r.t variables
	double* gradWRTlambda;			// Gradient of lagrangian w.r.t equality lagrange multipliers
	double* gradWRTsigma;			// Gradient of lagrangian w.r.t inequality lagrange multipliers
	double* gradWRTslack;			// Gradient of lagrangian w.r.t slack variables

	double* Hessian;				// Lagrangian Hessian matrix w.r.t variables
	double* JacWRTlambda;			// Lagrangian Jacobian matrix w.r.t lambda
	double* JacWRTsigma;			// Lagrangian Jacobian matrix w.r.t sigma
	double* DiagSlacks;				// Diagonal matrix with the second derivative of log(z) w.r.t z
	double* U;						// Matrix that contains the Hessians, jacobians and other info for Newton method solve
	double* r;						// Vector containing all the gradients needed for Newton method solve
	double* step;					// Vector returned by the householder solve
	double normStep;				// length of step vector
	size_t Usize;					// One Dimension of the square U matrix;
	size_t rSize;					// Number or rows in the R matrix;
} ConstraintInfo;


int InitializeConstrainInfo(ConstraintInfo* constInfo);
void FreeConstrainInfo(ConstraintInfo* constInfo);

void ComputeGradientLagrangianWRTvars(ConstraintInfo* constInfo);
void ComputeGradientLagrangianWRTslack(ConstraintInfo* constInfo);
void ComputeGradientLagrangianWRTlambda(ConstraintInfo* constInfo);
void ComputeGradientLagrangianWRTsigma(ConstraintInfo* constInfo);
void ComputeJacobianLagrangainWRTlambda(ConstraintInfo* constInfo);
void ComputeJacobianLagrangainWRTsigma(ConstraintInfo* constInfo);
void CreateMatrixU(ConstraintInfo* constInfo);
void CreateVecR(ConstraintInfo* constInfo);
void SolveStep(ConstraintInfo* constInfo);
void ApplyStep(ConstraintInfo* constInfo, double alpha);


int OptimizeConstrained(ConstraintInfo* constInfo);
void BfgsInverseUpdate(double* H, const double* s, const double* y, size_t n);
int Optimize_Steepest(EVALUATE eval, double* x0, int numVars, double stepLength, double* result);
int Optimize_QuasiNewton(EVALUATE eval, double* x0, int numVars, double* H0, double* result);
double CalculateStepLengthWeakWolfe(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars);
double CalculateStepLengthStrongWolfe(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars);
double CalculateStepLengthArmijo(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars);

static void PrintVector(const char* name, const double* vec, size_t size);
static void PrintMatrix(
	const char* name, const double* mat, size_t rows, size_t cols);