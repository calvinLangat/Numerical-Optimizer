#include "optimizer_const.h"


int InitializeConstrainInfo(ConstraintInfo* constInfo)
{
	int res = 0;

	size_t sizeofU = constInfo->numVarsFunc + constInfo->numInEqualityConst
		+ constInfo->numEqualityConst + constInfo->numInEqualityConst;
	constInfo->Usize = sizeofU * sizeofU;
	constInfo->rSize = sizeofU;

	constInfo->last_x = malloc(constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->last_x == NULL);

	constInfo->lambdas = malloc(constInfo->numEqualityConst * sizeof(double));
	res |= (constInfo->lambdas == NULL);

	constInfo->sigmas = malloc(constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->sigmas == NULL);

	constInfo->slacks = malloc(constInfo->numInEqualityConst * sizeof(double));
	res |= (constInfo->slacks == NULL);

	constInfo->gradWRTvars = malloc(constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->gradWRTvars == NULL);

	constInfo->last_gradWRTvars = malloc(constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->last_gradWRTvars == NULL);

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

	constInfo->s = malloc(constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->s == NULL);

	constInfo->y = malloc(constInfo->numVarsFunc * sizeof(double));
	res |= (constInfo->y == NULL);

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

	CreateIdentityMat(constInfo->Hessian, constInfo->numVarsFunc);

	for (size_t i = 0; i < sizeofU; ++i)
	    constInfo->step[i] = 0.0;

	constInfo->hasPreviousX = 0;
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

	free(constInfo->s);
	free(constInfo->y);

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

int ComputeLagrangianHessianWRTvars(ConstraintInfo* constInfo)
{
	const size_t n = constInfo->numVarsFunc;
    // First call: there is no previous displacement to learn from.
    if (!constInfo->hasPreviousX)
    {
        for (size_t i = 0; i < n; ++i)
            constInfo->last_x[i] = constInfo->x0[i];

        constInfo->hasPreviousX = 1;
        return 1;  // No update.
    }

    // Actual displacement between accepted points.
    VecSub(
        constInfo->x0,
        constInfo->last_x,
        constInfo->s,
        n);

    VecSub(
    constInfo->x0,
    constInfo->last_x,
    constInfo->s,
    n);

	double relativeStep = 0.0;

	for (size_t i = 0; i < n; ++i)
	{
	    const double scale =
	        fmax(1.0, fmax(fabs(constInfo->x0[i]),
	                      fabs(constInfo->last_x[i])));

	    const double change = fabs(constInfo->s[i]) / scale;

	    if (!isfinite(change))
	        return -2;

	    relativeStep = fmax(relativeStep, change);
	}

	// Starting value for current finite-difference implementation.
	const double minimumBfgsStep = 1e-6;

	if (relativeStep <= minimumBfgsStep)
	{
	    // Advance the saved point even though B stays unchanged.
	    for (size_t i = 0; i < n; ++i)
	        constInfo->last_x[i] = constInfo->x0[i];

	    return 1;  // BFGS update skipped; optimization continues.
	}

    // Save the current input/output pointers.
    double* currentX = constInfo->x0;
    double* currentGradient = constInfo->gradWRTvars;

    // Evaluate at the previous x, using CURRENT multipliers.
    // Reuse last_gradWRTvars as the output buffer.
    constInfo->x0 = constInfo->last_x;
    constInfo->gradWRTvars = constInfo->last_gradWRTvars;

    ComputeGradientLagrangianWRTvars(constInfo);

    // Restore the current input/output pointers.
    constInfo->x0 = currentX;
    constInfo->gradWRTvars = currentGradient;

    // Both gradients now use the same multipliers.
    VecSub(
        constInfo->gradWRTvars,
        constInfo->last_gradWRTvars,
        constInfo->y,
        n);

    int status = BfgsHessianUpdate(
        constInfo->Hessian,
        constInfo->s,
        constInfo->y,
        n);

    // Save the current point for the next accepted displacement.
    for (size_t i = 0; i < n; ++i)
        constInfo->last_x[i] = constInfo->x0[i];

    return status;
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

int CalculateStepLength(ConstraintInfo* constInfo)
{
    // Returns:
    //   0: accepted step
    //  -1: no acceptable step found
    //  -2: allocation failed
    //  -3: invalid input or current state

    const size_t n = constInfo->numVarsFunc;
    const size_t m = constInfo->numInEqualityConst;
    const size_t p = constInfo->numEqualityConst;
    const size_t N = constInfo->rSize;

    double alpha = constInfo->alpha;
    const double c1 = constInfo->c1;
    int status = -1;

    if (N == 0 || N != n + p + 2*m ||
        !isfinite(alpha) || alpha <= 0.0 || alpha > 1.0 ||
        !(c1 > 0.0 && c1 < 1.0) ||
        constInfo->maxArmijoTests <= 0)
    {
        constInfo->alpha = 0.0;
        return -3;
    }

    // Each entry points to an existing variable/gradient block.
    double* values[4] = {
        constInfo->x0,
        constInfo->slacks,
        constInfo->lambdas,
        constInfo->sigmas
    };

    double* gradients[4] = {
        constInfo->gradWRTvars,
        constInfo->gradWRTslack,
        constInfo->gradWRTlambda,
        constInfo->gradWRTsigma
    };

    const size_t sizes[4] = { n, m, p, m };
    const size_t offsets[4] = { 0, n, n + m, n + m + p };

    // Store backups in the same order as the Newton direction.
    double* originalValues = malloc(N * sizeof(double));
    double* originalGradients = malloc(N * sizeof(double));
    double* originalR = malloc(N * sizeof(double));

    if (!originalValues || !originalGradients || !originalR)
    {
        free(originalValues);
        free(originalGradients);
        free(originalR);

        constInfo->alpha = 0.0;
        return -2;
    }

    for (size_t block = 0; block < 4; ++block)
    {
        for (size_t j = 0; j < sizes[block]; ++j)
        {
            const size_t k = offsets[block] + j;

            originalValues[k] = values[block][j];
            originalGradients[k] = gradients[block][j];
        }
    }

    double M_current = 0.0;

    for (size_t i = 0; i < N; ++i)
    {
        originalR[i] = constInfo->r[i];
        M_current += originalR[i] * originalR[i];
    }

    M_current *= 0.5;

    if (!isfinite(M_current) || M_current <= 0.0)
    {
        // Convergence should be checked before calling this function.
        status = -3;
        goto restore;
    }

    for (size_t i = 0; i < N; ++i)
    {
        if (!isfinite(originalValues[i]) ||
            !isfinite(constInfo->step[i]))
        {
            status = -3;
            goto restore;
        }
    }

    for (size_t j = 0; j < m; ++j)
    {
        if (originalValues[n + j] <= 0.0 ||
            originalValues[n + m + p + j] <= 0.0)
        {
            status = -3;
            goto restore;
        }
    }

    // alpha arrives already capped for positivity.
    for (int attempt = 0;
         attempt < constInfo->maxArmijoTests;
         ++attempt)
    {
        int validTrial = 1;
        int changed = 0;

        // Every trial starts from the saved original point.
        for (size_t block = 0; block < 4; ++block)
        {
            for (size_t j = 0; j < sizes[block]; ++j)
            {
                const size_t k = offsets[block] + j;

                const double trial =
                    originalValues[k] + alpha * constInfo->step[k];

                values[block][j] = trial;

                if (!isfinite(trial))
                    validTrial = 0;

                // Blocks 1 and 3 are slacks and inequality multipliers.
                if ((block == 1 || block == 3) && !(trial > 0.0))
                    validTrial = 0;

                if (trial != originalValues[k])
                    changed = 1;
            }
        }

        // Smaller steps cannot help once the entire update rounds away.
        if (!changed)
            break;

        if (validTrial)
        {
            ComputeGradientLagrangianWRTvars(constInfo);
            ComputeGradientLagrangianWRTslack(constInfo);
            ComputeGradientLagrangianWRTlambda(constInfo);
            ComputeGradientLagrangianWRTsigma(constInfo);
            CreateVecR(constInfo);

            double M_trial = 0.0;

            for (size_t i = 0; i < N; ++i)
                M_trial += constInfo->r[i] * constInfo->r[i];

            M_trial *= 0.5;

            const double threshold =
                (1.0 - c1 * alpha) * M_current;

            if (isfinite(M_trial) &&
                M_trial < M_current &&
                M_trial <= threshold)
            {
                status = 0;
                break;
            }
        }

        alpha *= 0.7;

        if (alpha <= 0.0)
            break;
    }

restore:
    // This function selects alpha; the caller applies the step.
    for (size_t block = 0; block < 4; ++block)
    {
        for (size_t j = 0; j < sizes[block]; ++j)
        {
            const size_t k = offsets[block] + j;

            values[block][j] = originalValues[k];
            gradients[block][j] = originalGradients[k];
        }
    }

    for (size_t i = 0; i < N; ++i)
        constInfo->r[i] = originalR[i];

    constInfo->alpha = (status == 0) ? alpha : 0.0;

    free(originalValues);
    free(originalGradients);
    free(originalR);

    return status;
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
		// ComputeLagrangianHessianWRTvars(constInfo);
		// ComputeJacobianLagrangainWRTlambda(constInfo);
		// ComputeJacobianLagrangainWRTsigma(constInfo);
		// CreateMatrixU(constInfo);
		// CreateVecR(constInfo);
		// SolveStep(constInfo);
		// ApplyStep(constInfo);

		printf("Iteration: %d\n", iterations + 1);
		printf("mu = %.10g\n\n", constInfo->mu);

		ComputeGradientLagrangianWRTvars(constInfo);
		// PrintVector("gradWRTvars", constInfo->gradWRTvars,
		//     constInfo->numVarsFunc);

		ComputeGradientLagrangianWRTslack(constInfo);
		// PrintVector("gradWRTslack", constInfo->gradWRTslack,
		//     constInfo->numInEqualityConst);

		ComputeGradientLagrangianWRTlambda(constInfo);
		// PrintVector("gradWRTlambda", constInfo->gradWRTlambda,
		//     constInfo->numEqualityConst);

		ComputeGradientLagrangianWRTsigma(constInfo);
		//PrintVector("gradWRTsigma", constInfo->gradWRTsigma,
		//    constInfo->numInEqualityConst);

		ComputeLagrangianHessianWRTvars(constInfo);

		ComputeJacobianLagrangainWRTlambda(constInfo);
		//PrintMatrix("JacWRTlambda", constInfo->JacWRTlambda,
		//    constInfo->numEqualityConst, constInfo->numVarsFunc);

		ComputeJacobianLagrangainWRTsigma(constInfo);
		//PrintMatrix("JacWRTsigma", constInfo->JacWRTsigma,
		//    constInfo->numInEqualityConst, constInfo->numVarsFunc);

		CreateMatrixU(constInfo);
		//PrintMatrix("U", constInfo->U,
		//    constInfo->rSize, constInfo->rSize);

		// CreateVecR stores the negated residual for U * step = -r.
		CreateVecR(constInfo);
		//PrintVector("RHS (-r)", constInfo->r, constInfo->rSize);

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
		//PrintVector("step", constInfo->step, constInfo->rSize);

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

		// Pass the positivity-capped value into the line search.
		constInfo->alpha = alpha;

		int status = CalculateStepLength(constInfo);

		
		if (status != 0)
		    return status;

		ApplyStep(constInfo, constInfo->alpha);


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

// Returns:
//   0: updated
//   1: skipped because curvature was unsuitable
//  -1: allocation failed
int BfgsHessianUpdate(
    double* B,
    const double* s,
    const double* y,
    size_t n)
{
    double sy = 0.0;
    double normS = 0.0;
    double normY = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        sy += s[i] * y[i];
        normS = hypot(normS, s[i]);
        normY = hypot(normY, y[i]);
    }

    // Require positive curvature with a relative margin.
    const double curvatureTolerance = 1e-8;

    if (!isfinite(sy) ||
        !isfinite(normS) ||
        !isfinite(normY) ||
        normS == 0.0 ||
        normY == 0.0 ||
        sy <= curvatureTolerance * normS * normY)
    {
        return 1;
    }

    double* Bs = malloc(n * sizeof(*Bs));
    if (Bs == NULL)
        return -1;

    // Calculate B*s using the OLD matrix.
    for (size_t i = 0; i < n; ++i)
    {
        Bs[i] = 0.0;

        for (size_t j = 0; j < n; ++j)
            Bs[i] += B[i*n + j] * s[j];
    }

    double sBs = 0.0;

    for (size_t i = 0; i < n; ++i)
        sBs += s[i] * Bs[i];

    if (!isfinite(sBs) || sBs <= 0.0)
    {
        free(Bs);
        return 1;
    }

    // B_new = B - (Bs)(Bs)^T / (s^T Bs)
    //           + y*y^T / (s^T y)
    //
    // Update one triangle and mirror it to preserve symmetry.
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i; j < n; ++j)
        {
            const double value =
                B[i*n + j]
                - Bs[i] * Bs[j] / sBs
                + y[i] * y[j] / sy;

            B[i*n + j] = value;
            B[j*n + i] = value;
        }
    }

    free(Bs);
    return 0;
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
	// printf("%s (%zu x %zu):\n", name, rows, cols);
	// for (size_t i = 0; i < rows; ++i)
	// {
	// 	printf("  [");
	// 	for (size_t j = 0; j < cols; ++j)
	// 		printf("%s% .10g", j ? ", " : "", mat[i * cols + j]);
	// 	printf("]\n");
	// }
	// printf("\n");
}