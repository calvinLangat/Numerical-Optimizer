#include "utils.h"
#include "optimizer.h"

// Function to evaluate
double MyFunc(double* vals)
{
	double res = pow((1 - vals[0]), 2) + 100 * pow((vals[1] - vals[0] * vals[0]), 2);
	return res;
}

double MyFunc2(double* vals)
{
	double res = vals[0] * vals[0] + 2 * vals[1] + vals[1] * vals[1] + 5;
	return res;
}

double MyFunc3(double* vals)
{
	// Steepest Descent algorithm struggles with this function.
	// Time for a better Optimization Algo.


	//double fun = pow((50-3*tan(vals[0])),2) + pow((60-5*tan(vals[1])),2) + 2*(50-3*tan(vals[0]))*(60-5*tan(vals[1]))*cos(20/180*M_PI);

	double DEG_TO_RAD = M_PI / 180.0;
	double dist_A = 10.0;
	double dist_B = 10.0;
	double dist_E = 50.0;
	double dist_F = 50.0;
	double beta = 60.0;

	double tan_x1 = tan(vals[0]);
	double tan_x2 = tan(vals[1]);
	double cos_meetAngle = cos(beta * DEG_TO_RAD);

	double fun = pow((dist_E - dist_A * tan_x1), 2) + pow((dist_F - dist_B * tan_x2), 2) + 2 * (dist_E - dist_A * tan_x1) * (dist_F - dist_B * tan_x2) * cos_meetAngle;
	return sqrt(fun);
}

double MyFunc4(double* vals)
{
	return pow(vals[0],2) + pow(vals[1],2);
}

double MyFunc5(double* vals)
{
	return 100*pow(vals[0],2) + pow(vals[1],2);
}

double MyFunc6(double* vals)
{
	return pow(vals[0] - 3,2) + pow(vals[1] + 2,2);
}

double Rosenbrock(double* x)
{
    double a = 1.0 - x[0];
    double b = x[1] - x[0]*x[0];
    return a*a + 100.0*b*b;
}

double Himmelblau(double* x)
{
    double a = x[0]*x[0] + x[1] - 11.0;
    double b = x[0] + x[1]*x[1] - 7.0;
    return a*a + b*b;
}

double ExpValley(double* x)
{
    return exp(x[0] + 3.0*x[1] - 0.1)
         + exp(x[0] - 3.0*x[1] - 0.1)
         + exp(-x[0] - 0.1);
}

double SinRidge(double* x)
{
    return x[0]*x[0] + 100.0 * sin(x[1]) * sin(x[1]);
}

double PenaltyKink(double* x)
{
    double p = 0.0;
    if (x[0] > 1.0)
        p = x[0] - 1.0;
    return x[0]*x[0] + 100.0*p*p;
}

double QuadND(double* x)
{
    double sum = 0.0;
    for (int i = 0; i < 2; ++i)
        sum += (i + 1) * x[i] * x[i];
    return sum;
}

int main(int argc, char* argv[])
{
	
	// Initial conditions
	int 	 numVars = 2;
	EVALUATE objective = SinRidge;


	double* x0 = malloc(numVars * sizeof(double));
	double* x_res = malloc(numVars * sizeof(double));
	double* H0 = malloc(numVars * numVars * sizeof(double));	// Hessain
	memset(H0, 0, numVars * numVars * sizeof(double));

	x0[0] = 3.0;
	x0[1] = 3.0;

	// Start Hessian with an Identity matrix
	CreateIdentityMat(H0, numVars);

	int success = Optimize_QuasiNewton_FixedStep(objective, x0, numVars, H0, x_res);

	printf("Final Hessian:\n");
	for(int i=0; i<numVars; ++i)
	{
		for(int j=0; j<numVars; ++j)
		{
			printf("%f ",H0[i*numVars + j]);
		}
		printf("\n");
	}
	printf("\n");

	//int success = Optimize_Steepest(objective, x0, 2, 1E-7, x_res);
	printf("x*: %f, %f\n", x_res[0], x_res[1]);
	printf("Value at x*: %f\n", objective(x_res));


	free(x0);
	free(x_res);
	free(H0);
	return 0;
}