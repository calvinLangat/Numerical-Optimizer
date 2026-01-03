#include "optimizer.h"

void BfgsInverseUpdate(double* H, const double* s, const double* y, size_t n)
{
	double d = Dot(y, s, n);
	
	// Temporary since line search algorithm with Wolfe Conditions will handle this
	if (fabs(d) < 1E-12)
	{
		printf("BFGS: d too small: %f\n", d);
		return;
	}

	double rho = 1.0 / d;

	const size_t matSize = n * n;
	double* I_1 = malloc(matSize * sizeof(double));
	double* I_2 = malloc(matSize * sizeof(double));
	double* syT = malloc(matSize * sizeof(double));
	double* ysT = malloc(matSize * sizeof(double));
	double* ssT = malloc(matSize * sizeof(double));
	double* MatMultRes1 = malloc(matSize * sizeof(double));
	double* MatMultRes2 = malloc(matSize * sizeof(double));

	if (!I_1 || !I_2 || !syT || !ysT || !ssT || !MatMultRes1 || !MatMultRes2)
	{
		printf("Failed to malloc lol\n");
		return;
	}

	SetArrayZero(I_1, matSize);
	SetArrayZero(I_2, matSize);
	SetArrayZero(syT, matSize);
	SetArrayZero(ysT, matSize);
	SetArrayZero(ssT, matSize);
	SetArrayZero(MatMultRes1, matSize);
	SetArrayZero(MatMultRes2, matSize);
	CreateIdentityMat(I_1, n);
	CreateIdentityMat(I_2, n);

	// Compute outer products
	OuterProduct(s, y, n, syT);
	OuterProduct(y, s, n, ysT);
	OuterProduct(s, s, n, ssT);

	// Multiply by rho
	MatScalarMult(syT, rho, matSize);
	MatScalarMult(ysT, rho, matSize);
	MatScalarMult(ssT, rho, matSize);

	// Subtract from Identity
	MatSubSame(I_1, syT, matSize);
	MatSubSame(I_2, ysT, matSize);


	MatMul(I_1, H, MatMultRes1, n);
	MatMul(MatMultRes1, I_2, MatMultRes2, n);

	// Put final values for new Hessian into old Hessian
	MatAdd(MatMultRes2, ssT, H, matSize);
	
	free(I_1);
	free(I_2);
	free(syT);
	free(ysT);
	free(ssT);
	free(MatMultRes1);
	free(MatMultRes2);
}

// This is forward finite differencing. Less accurate than central differencing, but half the function calls
void ComputeGradient(EVALUATE eval, double* x0, int numVars, double* gradient)
{
	double* x = malloc(numVars * sizeof(double));
	double f;
	double* f_h = malloc(numVars * sizeof(double));
	double stepsize = 1E-8;

	for (int i = 0; i < numVars; ++i)
	{
		x[i] = x0[i];
	}

	f = eval(x);
	for (int i = 0; i < numVars; ++i)
	{
		x[i] += stepsize;
		f_h[i] = eval(x);

		x[i] -= stepsize;

		gradient[i] = (f_h[i] - f) / stepsize;
	}
	free(x);
	free(f_h);
}


int Optimize_Steepest(EVALUATE eval, double* x0, int numVars, double stepLength, double* result)
{
	// Simple steepest descent algorithm. No second derivatives (Hessians)
	// X_n+1 = X_n + h*p; p = -gradient

	double res = eval(x0);
	double* gradient_now = malloc(numVars * sizeof(double));
	double* x_n = malloc(numVars * sizeof(double));;
	double len = 0;
	int iterations = 0;

	for (int i = 0; i < numVars; ++i)
	{
		x_n[i] = x0[i];
	}

	// Step 1: get the gradient. Finite differencing for now
	ComputeGradient(eval, x0, numVars, gradient_now);
	printf("Initial Gradient: %f, %f\n", gradient_now[0], gradient_now[1]);
	printf("Initial Inputs: %f, %f\n", x_n[0], x_n[1]);
	printf("Initial result: %f\n", eval(x_n));

	len = CalcNorm2(gradient_now, numVars);

	// optimize until we reach our tolerance
	while (len > 1E-6)
	{
		for (int i = 0; i < numVars; ++i)
		{
			x_n[i] -= stepLength * gradient_now[i];
		}

		ComputeGradient(eval, x_n, numVars, gradient_now);
		len = CalcNorm2(gradient_now, numVars);

		iterations++;
		if(iterations % 1000 == 0)
		{
			printf("Iteration: %d\n", iterations);
			printf("Gradient now: %f, %f\n", gradient_now[0], gradient_now[1]);
			printf("x: %f, %f\n", x_n[0], x_n[1]);
			printf("Result at x: %f\n", eval(x_n));
			printf("Euclidian distance: %f\n\n", len);			
		}
	}

	for (int i = 0; i < numVars; ++i)
	{
		result[i] = x_n[i];
	}

	printf("Iterations: %d\n", iterations);
	
	free(gradient_now);
	free(x_n);
	return 0;
}

int Optimize_QuasiNewton_FixedStep(EVALUATE eval, double* x0, int numVars, double* H0, double* result)
{
	double res = eval(x0);
	double* gradient_now = malloc(numVars * sizeof(double));
	double* gradient_past = malloc(numVars * sizeof(double));
	double* x_n = malloc(numVars * sizeof(double));
	double* p = malloc(numVars * sizeof(double));
	double* x_n_past = malloc(numVars * sizeof(double));
	double* s = malloc(numVars * sizeof(double));
	double* y = malloc(numVars * sizeof(double));
	double len = 0;
	double stepLength = 1;
	int iterations = 0;

	SetArrayZero(s, numVars);
	SetArrayZero(y, numVars);
	SetArrayZero(x_n_past, numVars);
	SetArrayZero(gradient_past, numVars);


	// Step 1: get the gradient. Finite differencing for now
	ComputeGradient(eval, x0, numVars, gradient_now);
	len = CalcNorm2(gradient_now, numVars);
	
	for (int i = 0; i < numVars; ++i)
	{
		x_n[i] = x0[i];
		x_n_past[i] = x0[i];
		gradient_past[i] = gradient_now[i];
	}

	printf("Initial Inputs: %f, %f\n", x_n[0], x_n[1]);
	printf("Initial Gradient: %f, %f\n", gradient_now[0], gradient_now[1]);
	printf("Initial result: %f\n", eval(x_n));

	// NOTE(CL): We need to satisfy Wolfe conditions and also use the Hessain
	// NOTE(CL): Hessian Update done. Works but now I need to implement line Search Algorithm with Wolfe
	while (len > 1E-6)
	{
		// Compute step(fixed for now)
		MatVecMul(H0, gradient_now, p, numVars);
		VecScalarMult(p, -1, numVars);

		// Get step Length
		//stepLength = CalculateStepLengthArmijo(eval, x_n, p, gradient_now, numVars);
		//stepLength = CalculateStepLengthWeakWolfe(eval, x_n, p, gradient_now, numVars);
		stepLength = CalculateStepLengthStrongWolfe(eval, x_n, p, gradient_now, numVars);
		printf("Step Length: %f\n", stepLength);
		VecScalarMult(p, stepLength, numVars);
		VecAdd(x_n, p, numVars);

		// get gradient
		ComputeGradient(eval, x_n, numVars, gradient_now);
		len = CalcNorm2(gradient_now, numVars);

		// Calculate differences s_k and y_k
		VecSub(gradient_now, gradient_past, y, numVars);
		VecSub(x_n, x_n_past, s, numVars);

		// Update Hessian
		BfgsInverseUpdate(H0, s, y, numVars);

		// Store previous gradient and inputs
		for (int i = 0; i < numVars; ++i)
		{
			x_n_past[i] = x_n[i];
			gradient_past[i] = gradient_now[i];
		}

		if (iterations % 10 == 0)	// Print every 10 iterations
		{
			printf("Iteration: %d\n", iterations);
			printf("Gradient now: %f, %f\n", gradient_now[0], gradient_now[1]);
			printf("x: %f, %f\n", x_n[0], x_n[1]);
			printf("Result at x: %f\n", eval(x_n));
			printf("Euclidian distance: %f\n\n", len);
		}

		iterations++;
	}

	for (int i = 0; i < numVars; ++i)
	{
		result[i] = x_n[i];
	}
	printf("Iterations: %d\n", iterations);

	free(gradient_now);
	free(gradient_past);
	free(x_n);
	free(x_n_past);
	free(p);
	free(s);
	free(y);
	return 0;
}

double CalculateStepLengthStrongWolfe(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars)
{
	// Here we need to find the step step length. We already have the direction sorted.
	// To do this, we need to satisfy the Strong Wolfe condifions
	// 1. There must be a sufficient decrease in the function value.
	// 2. Curvature should be less than some metric.This enables us to skip the steepest bit and get near the bottom.

	double c1 = 1E-4;
	double c2 = 0.9;
	double minStep =  1E-4;
	double stepLength = 1.0; // Initial full step
	double candidateStepRes1 = 0.0;
	double candidateStepRes2 = 0.0;
	double candidateStepCondition1 = 0.0;
	double candidateStepCondition2 = 0.0;
	double* candidateStep = malloc(numVars * sizeof(double));
	double* candidateStepGrad = malloc(numVars * sizeof(double));
	double f0 = eval(x_n);

	SetArrayZero(candidateStep, numVars);
	SetArrayZero(candidateStepGrad, numVars);

	VecScalarMult2(direction, candidateStep, stepLength, numVars);
	VecAdd(candidateStep, x_n, numVars);
	
	double dotGradientDirection = Dot(gradient, direction, numVars);
	candidateStepRes1 = eval(candidateStep);
	candidateStepCondition1 = f0 + c1*stepLength*dotGradientDirection;
	ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);
	candidateStepRes2 = fabs(Dot(candidateStepGrad,direction, numVars));
	candidateStepCondition2 = c2 * fabs(dotGradientDirection);

	while(candidateStepRes1 > candidateStepCondition1 || candidateStepRes2 > candidateStepCondition2)
	{
		stepLength *= 0.65;
		stepLength = fmax(stepLength, minStep);
		VecScalarMult2(direction, candidateStep, stepLength, numVars);
		VecAdd(candidateStep, x_n, numVars);
		ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);

		candidateStepRes1 = eval(candidateStep);
		candidateStepCondition1 = f0 + c1*stepLength*dotGradientDirection;
		candidateStepRes2 = fabs(Dot(candidateStepGrad,direction, numVars));
		candidateStepCondition2 = c2 * fabs(dotGradientDirection);

		//printf("candidateStepRes1: %f\n", candidateStepRes1);
		//printf("candidateStepCondition1: %f\n", candidateStepCondition1);
		//printf("candidateStepRes2: %f\n", candidateStepRes2);
		//printf("candidateStepCondition2: %f\n", candidateStepCondition2);
		//printf("stepLength: %f\n", stepLength);
		
		if(stepLength < minStep)
			break;	// Couldn't find a reasonable stepLength (this check might not be needed)
	}

	free(candidateStep);
	free(candidateStepGrad);
	return stepLength;
}

double CalculateStepLengthWeakWolfe(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars)
{
	// Here we need to find the step step length. We already have the direction sorted.
	// To do this, we need to satisfy the weak Wolfe condifions
	// 1. There must be a sufficient decrease in the function value.
	// 2. Curvature should be less than some metric.This enables us to skip the steepest bit and get near the bottom.

	double c1 = 1E-4;
	double c2 = 0.9;
	double minStep =  1E-4;
	double stepLength = 1.0; // Initial full step
	double candidateStepRes1 = 0.0;
	double candidateStepRes2 = 0.0;
	double candidateStepCondition1 = 0.0;
	double candidateStepCondition2 = 0.0;
	double* candidateStep = malloc(numVars * sizeof(double));
	double* candidateStepGrad = malloc(numVars * sizeof(double));
	double f0 = eval(x_n);

	SetArrayZero(candidateStep, numVars);
	SetArrayZero(candidateStepGrad, numVars);

	VecScalarMult2(direction, candidateStep, stepLength, numVars);
	VecAdd(candidateStep, x_n, numVars);
	
	double dotGradientDirection = Dot(gradient, direction, numVars);
	candidateStepRes1 = eval(candidateStep);
	candidateStepCondition1 = f0 + c1*stepLength*dotGradientDirection;
	ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);
	candidateStepRes2 = Dot(candidateStepGrad,direction, numVars);
	candidateStepCondition2 = c2 * dotGradientDirection;

	while(candidateStepRes1 > candidateStepCondition1 || candidateStepRes2 < candidateStepCondition2)
	{
		stepLength *= 0.65;
		stepLength = fmax(stepLength, minStep);
		VecScalarMult2(direction, candidateStep, stepLength, numVars);
		VecAdd(candidateStep, x_n, numVars);
		ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);

		candidateStepRes1 = eval(candidateStep);
		candidateStepCondition1 = f0 + c1*stepLength*dotGradientDirection;
		candidateStepRes2 = Dot(candidateStepGrad,direction, numVars);
		candidateStepCondition2 = c2 * dotGradientDirection;

		//printf("candidateStepRes1: %f\n", candidateStepRes1);
		//printf("candidateStepCondition1: %f\n", candidateStepCondition1);
		//printf("candidateStepRes2: %f\n", candidateStepRes2);
		//printf("candidateStepCondition2: %f\n", candidateStepCondition2);
		//printf("stepLength: %f\n", stepLength);
		
		if(stepLength < minStep)
			break;	// Couldn't find a reasonable stepLength (this check might not be needed)
	}

	free(candidateStep);
	free(candidateStepGrad);
	return stepLength;
}

double CalculateStepLengthArmijo(EVALUATE eval, double* x_n, double* direction, double* gradient, int numVars)
{
	// Here we need to find the step step length. We already have the direction sorted.
	// To do this, we need to satisfy the Armijo condifions
	// 1. There must be a sufficient decrease in the function value.

	double c1 = 1E-4;
	double minStep =  1E-4;
	double stepLength = 1.0; // Initial full step
	double candidateStepRes1 = 0.0;
	double candidateStepCondition1 = 0.0;
	double* candidateStep = malloc(numVars * sizeof(double));
	double f0 = eval(x_n);

	SetArrayZero(candidateStep, numVars);

	VecScalarMult2(direction, candidateStep, stepLength, numVars);
	VecAdd(candidateStep, x_n, numVars);
	
	double dotGradientDirection = Dot(gradient, direction, numVars);

	if (dotGradientDirection >= 0.0)
	{
	    // Not a descent direction → fallback
	    free(candidateStep);
	    return 0.0;
	}

	candidateStepRes1 = eval(candidateStep);
	candidateStepCondition1 = f0 + c1*stepLength*dotGradientDirection;

	while(candidateStepRes1 > candidateStepCondition1)
	{
		stepLength *= 0.75; // half the step length
		stepLength = fmax(stepLength, minStep);
		VecScalarMult2(direction, candidateStep, stepLength, numVars);
		VecAdd(candidateStep, x_n, numVars);

		candidateStepRes1 = eval(candidateStep);
		candidateStepCondition1 = f0 + c1*stepLength*dotGradientDirection;

		//printf("candidateStepRes1: %f\n", candidateStepRes1);
		//printf("candidateStepCondition1: %f\n", candidateStepCondition1);
		//printf("stepLength: %f\n", stepLength);
		
		if(stepLength <= minStep)
			break;	// Couldn't find a reasonable stepLength (this check might not be needed)
	}

	free(candidateStep);
	return stepLength;

}