#include "../utils/utils.h"

typedef double (*EVALUATE)(double*);	// Function pointer signature for functions to be evaluated

typedef struct {
	EVALUATE func;					// Objective function
	EVALUATE* equalityConstFunc;	// Pointer to array of equality constraints function pointers
	EVALUATE* inEqualityConstFunc;	// Pointer to array of inequality constraints function pointers

	size_t numVarsFunc;				// Number of variables for the functions
	size_t numEqualityConst;		// Number of Equality constraints
	size_t numInEqualityConst;		// Number of Inequality constraints

	double* x0;						// Pointer to current x;
	double* last_x;					// Pointer to previous x;
	double* lambdas;				// Pointer to Lagrange multipliers for equality constraints
	double* sigmas;					// Pointer to Lagrange multipliers for inequality constraints
	double* slacks;					// Pointer to slack variables for inequality constraints
	double  mu;						// Penalty variable for log barrier slack variables

	double* gradWRTvars;			// Gradient of lagrangian w.r.t variables
	double* last_gradWRTvars;			// Previous Gradient of lagrangian w.r.t variables
	double* gradWRTlambda;			// Gradient of lagrangian w.r.t equality lagrange multipliers
	double* gradWRTsigma;			// Gradient of lagrangian w.r.t inequality lagrange multipliers
	double* gradWRTslack;			// Gradient of lagrangian w.r.t slack variables
	double* s;						// Change in vars;
	double* y;						// Change in gradient of vars;

	double* Hessian;				// Lagrangian Hessian matrix w.r.t variables
	double* JacWRTlambda;			// Lagrangian Jacobian matrix w.r.t lambda
	double* JacWRTsigma;			// Lagrangian Jacobian matrix w.r.t sigma
	double* DiagSlacks;				// Diagonal matrix with the second derivative of log(z) w.r.t z
	double* U;						// Matrix that contains the Hessians, jacobians and other info for Newton method solve
	double* r;						// Vector containing all the gradients needed for Newton method solve
	double* step;					// Vector returned by the householder solve
	double normStep;				// length of step vector
	double c1;						// Coeff for Armijo backtracking
	double alpha;					// Fraction of step to take
	size_t maxArmijoTests;			// Maximum number of iterations to find best step length
	size_t Usize;					// One Dimension of the square U matrix;
	size_t rSize;					// Number or rows in the R matrix;
	int hasPreviousX;
} ConstraintInfo;


int InitializeConstrainInfo(ConstraintInfo* constInfo);
void FreeConstrainInfo(ConstraintInfo* constInfo);

void ComputeGradientLagrangianWRTvars(ConstraintInfo* constInfo);
void ComputeGradientLagrangianWRTslack(ConstraintInfo* constInfo);
void ComputeGradientLagrangianWRTlambda(ConstraintInfo* constInfo);
void ComputeGradientLagrangianWRTsigma(ConstraintInfo* constInfo);
int  ComputeLagrangianHessianWRTvars(ConstraintInfo* constInfo);
void ComputeJacobianLagrangainWRTlambda(ConstraintInfo* constInfo);
void ComputeJacobianLagrangainWRTsigma(ConstraintInfo* constInfo);
void CreateMatrixU(ConstraintInfo* constInfo);
void CreateVecR(ConstraintInfo* constInfo);
void SolveStep(ConstraintInfo* constInfo);
int  CalculateStepLength(ConstraintInfo* constInfo);
void ApplyStep(ConstraintInfo* constInfo, double alpha);

int OptimizeConstrained(ConstraintInfo* constInfo);
int BfgsHessianUpdate(double* B, const double* s, const double* y, size_t n);

static void PrintVector(const char* name, const double* vec, size_t size);
static void PrintMatrix(
	const char* name, const double* mat, size_t rows, size_t cols);