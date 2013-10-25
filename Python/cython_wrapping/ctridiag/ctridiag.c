void ctridiag(double *a, double *b, double *c, double *x, int n){
    // Solve a tridiagonal system inplace.
    // Initialize temporary variable and index.
    int i;
    double temp;
    // Perform necessary computation in place.
    // Initial steps
    c[0] = c[0] / b[0];
    x[0] = x[0] / b[0];
    // Iterate down arrays.
    for (i=0; i<n-2; i++){
        temp = 1.0 / (b[i+1] - a[i] * c[i]);
        c[i+1] = c[i+1] * temp;
        x[i+1] = (x[i+1] - a[i] * x[i]) * temp;
        }
    // Perform last step.
    x[n-1] = (x[n-1] - a[n-2] * x[n-2]) / (b[n-1] - a[n-2] * c[n-2]);
    // Perform back substitution to finish constructing the solution.
    for (i=n-2; i>-1; i--){
        x[i] = x[i] - c[i] * x[i+1];
        }
    }
