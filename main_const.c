#include "utils.h"
#include "optimizer_const.h"

// Function to evaluate
double Objective(double* vals)
{
	double res = pow((vals[0] - 2), 2) + pow((vals[1]-1), 2);
	return res;
}

double EqConstraint1(double* vals)
{
	double res = vals[0] + vals[1] - 2;
	return res;
}

double IneqConstraint1(double* vals)
{
	double res = vals[0] - 0.5;
	return res;
}

int main(int argc, char* argv[])
{
	int numEq = 1;
	int numIneq = 1;
	int numVars = 2;

	ConstraintInfo constraintProblem = {0};
	constraintProblem.func = Objective;
	constraintProblem.numVarsFunc = numVars;
	constraintProblem.x0 = malloc(numVars*sizeof(double));

	constraintProblem.equalityConstFunc = malloc(numEq*sizeof(EVALUATE));
	constraintProblem.inEqualityConstFunc = malloc(numIneq*sizeof(EVALUATE));
	constraintProblem.equalityConstFunc[0] = EqConstraint1;
	constraintProblem.inEqualityConstFunc[0] = IneqConstraint1;
	constraintProblem.numEqualityConst = numEq;
	constraintProblem.numInEqualityConst = numIneq;

	int result = InitializeConstrainInfo(&constraintProblem);
	if(result != 0)
		printf("Failed to initialize constraintInfo struct");

	constraintProblem.x0[0] = 0.0;
	constraintProblem.x0[1] = 2.0;

	constraintProblem.Hessian[0] = 2.0;
	constraintProblem.Hessian[1] = 0.0;
	constraintProblem.Hessian[2] = 0.0;
	constraintProblem.Hessian[3] = 2.0;
	constraintProblem.mu = 1;


	result = OptimizeConstrained(&constraintProblem);

	if (result == 0)
	{
	    printf("Solution: %.10g, %.10g\n",
	           constraintProblem.x0[0],
	           constraintProblem.x0[1]);
	}
	else
	{
	    printf("Optimization failed: %d\n", result);
	}

	
	FreeConstrainInfo(&constraintProblem);
	free(constraintProblem.x0);
	free(constraintProblem.equalityConstFunc);
	free(constraintProblem.inEqualityConstFunc);
	return 0;
}