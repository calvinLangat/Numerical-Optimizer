#include "../utils/utils.h"
#include "optimizer_const.h"

// Function to evaluate
double Objective(double* vals)
{
    double a = vals[0] - 1.5;
    double b = vals[1] - 2.75;
    double c = vals[2] - 0.25;
    double d = vals[3] - 1.875;

    return a*a + b*b + c*c + d*d;
}

// Nonlinear equality.
double EqConstraint1(double* vals)
{
    return vals[0] * vals[1] - vals[2] - 1.0;
}

// Linear equality.
double EqConstraint2(double* vals)
{
    return vals[0] + vals[1] + vals[2] + vals[3] - 6.0;
}

// Circle bound.
double IneqConstraint1(double* vals)
{
    return vals[0]*vals[0] + vals[1]*vals[1] - 5.0;
}

// vals[2] >= 0.25.
double IneqConstraint2(double* vals)
{
    return 0.25 - vals[2];
}

// vals[3] <= 2.
double IneqConstraint3(double* vals)
{
    return vals[3] - 2.0;
}

int main(int argc, char* argv[])
{
	int numVars = 4;
	int numEq = 2;
	int numIneq = 3;

	ConstraintInfo constraintProblem = {0};
	
	constraintProblem.numVarsFunc = numVars;
	constraintProblem.numEqualityConst = numEq;
	constraintProblem.numInEqualityConst = numIneq;

	constraintProblem.func = Objective;

	constraintProblem.x0 = malloc(numVars*sizeof(double));
	constraintProblem.equalityConstFunc = malloc(numEq*sizeof(EVALUATE));
	constraintProblem.inEqualityConstFunc = malloc(numIneq*sizeof(EVALUATE));
	
	constraintProblem.equalityConstFunc[0] = EqConstraint1;
	constraintProblem.equalityConstFunc[1] = EqConstraint2;
	constraintProblem.inEqualityConstFunc[0] = IneqConstraint1;
	constraintProblem.inEqualityConstFunc[1] = IneqConstraint2;
	constraintProblem.inEqualityConstFunc[2] = IneqConstraint3;
	

	int result = InitializeConstrainInfo(&constraintProblem);
	if(result != 0)
		printf("Failed to initialize constraintInfo struct");

	constraintProblem.x0[0] = 1.5;
	constraintProblem.x0[1] = 1.5;
	constraintProblem.x0[2] = 1.25;
	constraintProblem.x0[3] = 1.75;
	
	constraintProblem.mu = 1;
	constraintProblem.maxArmijoTests = 40;
	constraintProblem.c1 = 1e-5;
	constraintProblem.alpha = 1;


	result = OptimizeConstrained(&constraintProblem);

	if (result == 0)
	{
	    printf("Solution: "); 
	    for(int i=0;i<constraintProblem.numVarsFunc;++i)
	    {
	    	printf("%.10g, ",constraintProblem.x0[i]);
	    }
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