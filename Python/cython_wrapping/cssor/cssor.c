void cssor(double* U, int m, int n, double omega, double tol, int maxiters, int* info){
    // info is passed as a pointer so the function can modify it as needed.
    // Temporary variables:
    // 'maxerr' is a temporary value.
    // It is used to determine when to stop iteration.
    // 'i', 'j', and 'k' are indices for the loops.
    // lcf and rcf will be precomputed values
    // used on the inside of the loops.
    double maxerr, temp, lcf, rcf;
    int i, j, k;
    lcf = 1.0 - omega;
    rcf = 0.25 * omega;
    for (k=0; k<maxiters; k++){
        maxerr = 0.0;
        for (j=1; j<n-1; j++){
            for (i=1; i<m-1; i++){
                temp = U[i*n+j];
                U[i*n+j] = lcf * U[i*n+j] + rcf * (U[i*n+j-1] + U[i*n+j+1] + U[(i-1)*n+j] + U[(i+1)*n+j]);
                maxerr = fmax(fabs(U[i*n+j] - temp), maxerr);}}
        // Break the outer loop if within
        // the desired tolerance.
        if (maxerr < tol){break;}}
    // Here we have it set status to 0 if
    // the desired tolerance was attained
    // within the the given maximum
    // number of iterations.
    if (maxerr < tol){*info=0;}
    else{*info=1;}}
