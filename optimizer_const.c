#include "optimizer_const.h"


int InitializeConstrainInfo(ConstraintInfo* constInfo)
{
	int res = 0;

	size_t sizeofU = constInfo->numVarsFunc + constInfo->numInEqualityConst
		+ constInfo->numEqualityConst + constInfo->numInEqualityConst;
	constInfo->Usize = sizeofU * sizeofU;
	constInfo->rSize = sizeofU;

	constInfo->lambdas = malloc(constInfo->numEqualityConst * sizeof(double));
	res |= (constInfo->lambdas == NULL);

	constInfo->sigmas = malloc(constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->sigmas == NULL);

	constInfo->slacks = malloc(constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->slacks == NULL);

	constInfo->gradWRTvars = malloc(constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->gradWRTvars == NULL);

	constInfo->gradWRTlambda = malloc(constInfo->numEqualityConst * sizeof(double));
	res |= (constInfo->gradWRTlambda == NULL);

	constInfo->gradWRTsigma = malloc(constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->gradWRTsigma == NULL);

	constInfo->gradWRTslack = malloc(constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->gradWRTslack == NULL);

	constInfo->Hessian = malloc(constInfo->numVarsFunc * constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->Hessian == NULL);

	constInfo->JacWRTlambda = malloc(constInfo->numEqualityConst * constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->JacWRTlambda == NULL);

	constInfo->JacWRTsigma = malloc(constInfo->numInEqualityConst * constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->JacWRTsigma == NULL);

	constInfo->DiagSlacks = malloc(constInfo->numInEqualityConst * constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->DiagSlacks == NULL);

	constInfo->U = malloc(sizeofU * sizeofU * sizeof(double));
	res |= (constInfo->U == NULL);

	constInfo->r = malloc(sizeofU * sizeof(double));
	res |= (constInfo->r == NULL);

	constInfo->step = malloc(sizeofU * sizeof(double));
	res |= (constInfo->step == NULL);


	for (size_t i = 0; i < constInfo->numEqualityConst; ++i)
    constInfo->lambdas[i] = 0.0;

	for (size_t i = 0; i < constInfo->numInEqualityConst; ++i)
	{
	    constInfo->sigmas[i] = 1.0;
	    constInfo->slacks[i] = 0.5;
	}

	for (size_t i = 0; i < sizeofU; ++i)
	    constInfo->step[i] = 0.0;


	return res;
}

void FreeConstrainInfo(ConstraintInfo* constInfo)
{
	if (constInfo == NULL)
		return;

	free(constInfo->lambdas);
	free(constInfo->sigmas);
	free(constInfo->slacks);

	free(constInfo->gradWRTvars);
	free(constInfo->gradWRTlambda);
	free(constInfo->gradWRTsigma);
	free(constInfo->gradWRTslack);

	free(constInfo->Hessian);
	free(constInfo->JacWRTlambda);
	free(constInfo->JacWRTsigma);
	free(constInfo->DiagSlacks);
	free(constInfo->U);

	free(constInfo->r);
	free(constInfo->step);

	constInfo->lambdas = NULL;
	constInfo->sigmas = NULL;
	constInfo->slacks = NULL;

	constInfo->gradWRTvars = NULL;
	constInfo->gradWRTlambda = NULL;
	constInfo->gradWRTsigma = NULL;
	constInfo->gradWRTslack = NULL;

	constInfo->Hessian = NULL;
	constInfo->JacWRTlambda = NULL;
	constInfo->JacWRTsigma = NULL;
	constInfo->DiagSlacks = NULL;
	constInfo->U = NULL;

	constInfo->r = NULL;
	constInfo->step = NULL;

	constInfo->Usize = 0;
	constInfo->rSize = 0;
}


// This is forward finite differencing. Less accurate than central differencing, but half the function calls
void ComputeGradientLagrangianWRTvars(ConstraintInfo* constInfo)
{
	double* x = malloc(constInfo->numVarsFunc * sizeof(double));
	double f;
	double* f_h = malloc(constInfo->numVarsFunc * sizeof(double));
	double stepsize = 1E-8;

	// Keep local copies
	for (int i = 0; i < constInfo->numVarsFunc; ++i)
	{
		x[i] = constInfo->x0[i];
	}
	
	// Evaluate func at current x;
	f = constInfo->func(x);

	///////////////////////////////
	// Step 1
	///////////////////////////////
	// Gradient of objective func
	for(int i=0; i < constInfo->numVarsFunc; ++i )
	{
		double original = x[i];
		x[i] += stepsize;
		f_h[i] = constInfo->func(x);
		x[i] = original;

		constInfo->gradWRTvars[i] = (f_h[i]-f) / stepsize;
	}

	///////////////////////////////
	// Step 2
	///////////////////////////////
	// Gradient of equality constraints, have to matrix-vector mul to get a vector from the jacobian
	// jacobian matrix size is m x n, m = num of eqality constraints and n is the number of variables
	// WARN: Careful of transpose
	double* J_h = malloc(constInfo->numVarsFunc * constInfo->numEqualityConst * sizeof(double));
	double* vecH = malloc(constInfo->numVarsFunc * sizeof(double));
	EVALUATE EqConstFunc;

	for(int i=0;i<constInfo->numEqualityConst;++i)
	{
		EqConstFunc = constInfo->equalityConstFunc[i];
		// Evaluate equality constraint at current x;
		f = EqConstFunc(x);
		for(int j=0;j<constInfo->numVarsFunc;++j)
		{
			double original = x[j];
			x[j] += stepsize;
			f_h[j] = EqConstFunc(x);
			x[j] = original;

			J_h[i*constInfo->numVarsFunc+j] = (f_h[j]-f) / stepsize;
		}	
	}

	// Multiply jacobian(transposed) with the lambdas
	MatVecMulTransposeRect(J_h, constInfo->lambdas, vecH,  constInfo->numEqualityConst, constInfo->numVarsFunc);

	///////////////////////////////
	// Step 3
	///////////////////////////////
	// Gradient of inequality constraints, have to matrix-vector mul to get a vector from the jacobian
	// jacobian matrix size is p x n, p = num of ineqality constraints and n is the number of variables
	// WARN: Careful of transpose
	double* J_g = malloc(constInfo->numVarsFunc * constInfo->numInEqualityConst * sizeof(double));
	double* vecG = malloc(constInfo->numVarsFunc * sizeof(double));
	EVALUATE IneqConstFunc;

	// Evaluate inequality constraint at current x
	for(int i=0;i<constInfo->numInEqualityConst;++i)
	{
		IneqConstFunc = constInfo->inEqualityConstFunc[i];
		// Evaluate inequality constraint at current x;
		f = IneqConstFunc(x);
		for(int j=0;j<constInfo->numVarsFunc;++j)
		{
			double original = x[j];
			x[j] += stepsize;
			f_h[j] = IneqConstFunc(x);
			x[j] = original;

			J_g[i*constInfo->numVarsFunc+j] = (f_h[j] - f) / stepsize;
		}	
	}

	// Multiply jacobian(transposed) with the lambdas
	MatVecMulTransposeRect(J_g, constInfo->sigmas, vecG, constInfo->numInEqualityConst, constInfo->numVarsFunc);

	///////////////////////////////
	// Step 4
	///////////////////////////////
	// Add the three vectors together and store in 'gradWRTvars'
	VecAdd(constInfo->gradWRTvars, vecH, constInfo->numVarsFunc);
	VecAdd(constInfo->gradWRTvars, vecG, constInfo->numVarsFunc);

	free(x); 
	free(f_h);
	free(J_h);
	free(J_g);
	free(vecH);
	free(vecG);
}

void ComputeGradientLagrangianWRTslack(ConstraintInfo* constInfo)
{
	for(int i=0;i<constInfo->numInEqualityConst;++i)
	{
		constInfo->gradWRTslack[i] = constInfo->sigmas[i] - constInfo->mu/constInfo->slacks[i];
	}
}

void ComputeGradientLagrangianWRTlambda(ConstraintInfo* constInfo)
{
	EVALUATE EqConstFunc;
	double f;
	for(int i=0;i<constInfo->numEqualityConst;++i)
	{
		EqConstFunc = constInfo->equalityConstFunc[i];
		// Evaluate equality constraint at current x;
		f = EqConstFunc(constInfo->x0);
		constInfo->gradWRTlambda[i] = f;
	}
}

void ComputeGradientLagrangianWRTsigma(ConstraintInfo* constInfo)
{
	EVALUATE IneqConstFunc;
	double f;
	for(int i=0;i<constInfo->numInEqualityConst;++i)
	{
		IneqConstFunc = constInfo->inEqualityConstFunc[i];
		// Evaluate inequality constraint at current x;
		f = IneqConstFunc(constInfo->x0);
		constInfo->gradWRTsigma[i] = f + constInfo->slacks[i];
	}
}

void ComputeJacobianLagrangainWRTlambda(ConstraintInfo* constInfo)
{
	EVALUATE EqConstFunc;
	double* x = malloc(constInfo->numVarsFunc * sizeof(double));
	double f;
	double f_h;
	double stepsize = 1E-8;

	// Take local copy of x's
	for(int i=0; i<constInfo->numVarsFunc; ++i)
	{
		x[i] = constInfo->x0[i];
	}

	// Go through all constraints
	for(int i=0; i<constInfo->numEqualityConst; ++i)
	{
		EqConstFunc = constInfo->equalityConstFunc[i];
		f = EqConstFunc(x);

		// Compute gradient for each constraint
		for(int j=0; j<constInfo->numVarsFunc;++j)
		{
			double original = x[j];
			x[j] += stepsize;
			f_h = EqConstFunc(x);
			x[j] = original;

			constInfo->JacWRTlambda[i*constInfo->numVarsFunc+j] = (f_h-f)/stepsize;
		}
	}

	free(x);
}

void ComputeJacobianLagrangainWRTsigma(ConstraintInfo* constInfo)
{
	EVALUATE InEqConstFunc;
	double* x = malloc(constInfo->numVarsFunc * sizeof(double));
	double f;
	double f_h;
	double stepsize = 1E-8;

	// Take local copy of x's
	for(int i=0; i<constInfo->numVarsFunc; ++i)
	{
		x[i] = constInfo->x0[i];
	}

	// Go through all constraints
	for(int i=0; i<constInfo->numInEqualityConst; ++i)
	{
		InEqConstFunc = constInfo->inEqualityConstFunc[i];
		f = InEqConstFunc(x);

		// Compute gradient for each constraint
		for(int j=0; j<constInfo->numVarsFunc;++j)
		{
			double original = x[j];
			x[j] += stepsize;
			f_h = InEqConstFunc(x);
			x[j] = original;

			constInfo->JacWRTsigma[i*constInfo->numVarsFunc+j] = (f_h-f)/stepsize;
		}
	}

	free(x);
}

void CreateMatrixU(ConstraintInfo* constInfo)
{
    const size_t n = constInfo->numVarsFunc;
    const size_t p = constInfo->numEqualityConst;
    const size_t m = constInfo->numInEqualityConst;
    const size_t N = n + p + 2*m;

    // Starting row/column of each variable block.
    const size_t zOffset      = n;
    const size_t lambdaOffset = n + m;
    const size_t sigmaOffset  = n + m + p;

    double* U = constInfo->U;

    // U must already have space for N*N doubles.
    for (size_t row = 0; row < N; ++row)
    {
        for (size_t col = 0; col < N; ++col)
        {
            U[row*N + col] = 0.0;
        }
    }

    // Hessian: n rows, n columns.
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            U[i*N + j] = constInfo->Hessian[i*n + j];
        }
    }

    // Equality Jacobian and its transpose.
    for (size_t j = 0; j < p; ++j)       // Constraint
    {
        for (size_t i = 0; i < n; ++i)   // Variable
        {
            const double a = constInfo->JacWRTlambda[j*n + i];

            U[(lambdaOffset + j)*N + i] = a;
            U[i*N + (lambdaOffset + j)] = a;
        }
    }

    // Inequality Jacobian and its transpose.
    for (size_t j = 0; j < m; ++j)       // Constraint
    {
        for (size_t i = 0; i < n; ++i)   // Variable
        {
            const double a = constInfo->JacWRTsigma[j*n + i];

            U[(sigmaOffset + j)*N + i] = a;
            U[i*N + (sigmaOffset + j)] = a;
        }
    }

    // Slack curvature and the two identity blocks.
    for (size_t i = 0; i < m; ++i)
    {
        const size_t zRow = zOffset + i;
        const size_t sigmaRow = sigmaOffset + i;
        const double z = constInfo->slacks[i];  // Must be > 0.

        U[zRow*N + zRow] = constInfo->mu / (z*z);

        U[zRow*N + sigmaRow] = 1.0;
        U[sigmaRow*N + zRow] = 1.0;
    }
}

void CreateVecR(ConstraintInfo* constInfo)
{
	size_t offset=0;
	
	// Copy Gradient of lagrangian w.r.t variables
	for(int i=0; i<constInfo->numVarsFunc;++i)
	{
		constInfo->r[offset+i] = constInfo->gradWRTvars[i];
	}
	offset += constInfo->numVarsFunc;

	// Copy Gradient of lagrangian w.r.t slack variables
	for(int i=0; i<constInfo->numInEqualityConst;++i)
	{
		constInfo->r[offset+i] = constInfo->gradWRTslack[i];
	}
	offset += constInfo->numInEqualityConst;

	// Copy Gradient of lagrangian w.r.t equality lagrange multipliers
	for(int i=0; i<constInfo->numEqualityConst;++i)
	{
		constInfo->r[offset+i] = constInfo->gradWRTlambda[i];
	}
	offset += constInfo->numEqualityConst;

	// Copy Gradient of lagrangian w.r.t inequality lagrange multipliers
	for(int i=0; i<constInfo->numInEqualityConst;++i)
	{
		constInfo->r[offset+i] = constInfo->gradWRTsigma[i];
	}

	// Has to be in the form Ux=-r
	VecScalarMult(constInfo->r, -1, constInfo->rSize);
}

void SolveStep(ConstraintInfo* constInfo)
{
	/*
	 * Returns:
	 *   0  success
	 *  -1  invalid arguments
	 *  -2  allocation failed
	 *  -3  singular matrix
	 */

	int result = Householder_solve(constInfo->U, constInfo->r, constInfo->step, constInfo->rSize);
	switch (result)
	{
	case -1:
		printf("Householder solve: invalid arguments");
		break;
	case -2:
		printf("Householder solve: allocation failed");
		break;
	case -3:
		printf("Householder solve: singular matrix");
		break;
	}
}

void ApplyStep(ConstraintInfo* constInfo, double alpha)
{
    const size_t n = constInfo->numVarsFunc;
    const size_t m = constInfo->numInEqualityConst;
    const size_t p = constInfo->numEqualityConst;

    for (size_t i = 0; i < n; ++i)
        constInfo->x0[i] += alpha * constInfo->step[i];

    for (size_t i = 0; i < m; ++i)
        constInfo->slacks[i] += alpha * constInfo->step[n + i];

    for (size_t i = 0; i < p; ++i)
        constInfo->lambdas[i] += alpha * constInfo->step[n + m + i];

    for (size_t i = 0; i < m; ++i)
        constInfo->sigmas[i] += alpha * constInfo->step[n + m + p + i];
}

int OptimizeConstrained(ConstraintInfo* constInfo)
{
	double len = 0;
	int iterations = 0;
	const double residualTolerance = 1e-6;
	const double muTarget = 1e-8;
	const int maxIterations = 500;

	// Optimization loop
	for (iterations = 0; iterations < maxIterations; ++iterations)
	{
		// ComputeGradientLagrangianWRTvars(constInfo);
		// ComputeGradientLagrangianWRTslack(constInfo);
		// ComputeGradientLagrangianWRTlambda(constInfo);
		// ComputeGradientLagrangianWRTsigma(constInfo);
		// ComputeJacobianLagrangainWRTlambda(constInfo);
		// ComputeJacobianLagrangainWRTsigma(constInfo);
		// CreateMatrixU(constInfo);
		// CreateVecR(constInfo);
		// SolveStep(constInfo);
		// ApplyStep(constInfo);

		printf("Iteration: %d\n", iterations + 1);
		printf("mu = %.10g\n\n", constInfo->mu);

		ComputeGradientLagrangianWRTvars(constInfo);
		PrintVector("gradWRTvars", constInfo->gradWRTvars,
		    constInfo->numVarsFunc);

		ComputeGradientLagrangianWRTslack(constInfo);
		PrintVector("gradWRTslack", constInfo->gradWRTslack,
		    constInfo->numInEqualityConst);

		ComputeGradientLagrangianWRTlambda(constInfo);
		PrintVector("gradWRTlambda", constInfo->gradWRTlambda,
		    constInfo->numEqualityConst);

		ComputeGradientLagrangianWRTsigma(constInfo);
		PrintVector("gradWRTsigma", constInfo->gradWRTsigma,
		    constInfo->numInEqualityConst);

		ComputeJacobianLagrangainWRTlambda(constInfo);
		PrintMatrix("JacWRTlambda", constInfo->JacWRTlambda,
		    constInfo->numEqualityConst, constInfo->numVarsFunc);

		ComputeJacobianLagrangainWRTsigma(constInfo);
		PrintMatrix("JacWRTsigma", constInfo->JacWRTsigma,
		    constInfo->numInEqualityConst, constInfo->numVarsFunc);

		CreateMatrixU(constInfo);
		PrintMatrix("U", constInfo->U,
		    constInfo->rSize, constInfo->rSize);

		// CreateVecR stores the negated residual for U * step = -r.
		CreateVecR(constInfo);
		PrintVector("RHS (-r)", constInfo->r, constInfo->rSize);

		// Maximum absolute residual component.
		// The minus sign in the stored RHS does not affect this.
		double residualMax = 0.0;

		for (size_t i = 0; i < constInfo->rSize; ++i)
		{
		    const double value = fabs(constInfo->r[i]);

		    if (!isfinite(value))
		        return -2;

		    if (value > residualMax)
		        residualMax = value;
		}

		printf("Maximum absolute residual = %.10g\n", residualMax);

		if (residualMax <= residualTolerance)
		{
		    if (constInfo->mu <= muTarget)
		        return 0;  // Final barrier subproblem solved.

		    constInfo->mu = fmax(muTarget, 0.2 * constInfo->mu);

		    printf("Reducing mu to %.10g\n\n", constInfo->mu);

		    // Recompute residuals and U using the new mu.
		    // Keep the current x, slacks and multipliers.
		    continue;
		}

		SolveStep(constInfo);
		PrintVector("step", constInfo->step, constInfo->rSize);

		len = CalcNorm2(constInfo->step, constInfo->rSize);
		constInfo->normStep = len;
		printf("Newton step norm = %.10g\n\n", constInfo->normStep);


		double alpha = 1.0;
		const double fraction = 0.995;

		const size_t n = constInfo->numVarsFunc;
		const size_t m = constInfo->numInEqualityConst;
		const size_t p = constInfo->numEqualityConst;

		for (size_t i = 0; i < m; ++i)
		{
		    const double dz = constInfo->step[n + i];
		    const double dsigma = constInfo->step[n + m + p + i];

		    if (dz < 0.0)
		    {
		        const double limit =
		            -fraction * constInfo->slacks[i] / dz;

		        if (limit < alpha)
		            alpha = limit;
		    }

		    if (dsigma < 0.0)
		    {
		        const double limit =
		            -fraction * constInfo->sigmas[i] / dsigma;

		        if (limit < alpha)
		            alpha = limit;
		    }
		}


		ApplyStep(constInfo, alpha);

		PrintVector("x after update", constInfo->x0,
		    constInfo->numVarsFunc);

		PrintVector("slacks after update", constInfo->slacks,
		    constInfo->numInEqualityConst);

		PrintVector("lambdas after update", constInfo->lambdas,
		    constInfo->numEqualityConst);

		PrintVector("sigmas after update", constInfo->sigmas,
		    constInfo->numInEqualityConst);

		printf("f(x) after update = %.10g\n\n",
		    constInfo->func(constInfo->x0));
	}
	
	return -1;  // Iteration limit reached.
}


int Optimize_Steepest(EVALUATE eval, double* x0, int numVars, double stepLength, double* result)
{
	// Simple steepest descent algorithm. No second derivatives (Hessians)
	// X_n+1 = X_n + h*p; p = -gradient

	double res = eval(x0);
	double* gradient_now = malloc(numVars * sizeof(double));
	double* x_n = malloc(numVars * sizeof(double));
	double len = 0;
	int iterations = 0;

	for (int i = 0; i < numVars; ++i)
	{
		x_n[i] = x0[i];
	}

	// Step 1: get the gradient. Finite differencing for now
	//ComputeGradient(eval, x0, numVars, gradient_now);
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

		//ComputeGradient(eval, x_n, numVars, gradient_now);
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

int Optimize_QuasiNewton(EVALUATE eval, double* x0, int numVars, double* H0, double* result)
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
	//ComputeGradient(eval, x0, numVars, gradient_now);
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
		//ComputeGradient(eval, x_n, numVars, gradient_now);
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
	//ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);
	candidateStepRes2 = fabs(Dot(candidateStepGrad,direction, numVars));
	candidateStepCondition2 = c2 * fabs(dotGradientDirection);

	while(candidateStepRes1 > candidateStepCondition1 || candidateStepRes2 > candidateStepCondition2)
	{
		stepLength *= 0.65;
		stepLength = fmax(stepLength, minStep);
		VecScalarMult2(direction, candidateStep, stepLength, numVars);
		VecAdd(candidateStep, x_n, numVars);
		//ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);

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
	//ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);
	candidateStepRes2 = Dot(candidateStepGrad,direction, numVars);
	candidateStepCondition2 = c2 * dotGradientDirection;

	while(candidateStepRes1 > candidateStepCondition1 || candidateStepRes2 < candidateStepCondition2)
	{
		stepLength *= 0.65;
		stepLength = fmax(stepLength, minStep);
		VecScalarMult2(direction, candidateStep, stepLength, numVars);
		VecAdd(candidateStep, x_n, numVars);
		//ComputeGradient(eval, candidateStep, numVars, candidateStepGrad);

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


static void PrintVector(const char* name, const double* vec, size_t size)
{
	printf("%s = [", name);
	for (size_t i = 0; i < size; ++i)
		printf("%s%.10g", i ? ", " : "", vec[i]);
	printf("]\n\n");
}

static void PrintMatrix(
	const char* name, const double* mat, size_t rows, size_t cols)
{
	printf("%s (%zu x %zu):\n", name, rows, cols);
	for (size_t i = 0; i < rows; ++i)
	{
		printf("  [");
		for (size_t j = 0; j < cols; ++j)
			printf("%s% .10g", j ? ", " : "", mat[i * cols + j]);
		printf("]\n");
	}
	printf("\n");
}