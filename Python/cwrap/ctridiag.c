void ctridiag(double *a, double *b, double *c, double *x, int n){
    int i;
    double temp;
    c[0] = c[0] / b[0];
    x[0] = x[0] / b[0];
    for (i=0;i<n-2;i++){
        temp = 1.0 / (b[i+1]-a[i]*c[i]);
        c[i+1] = c[i+1] * temp;
        x[i+1] = (x[i+1]-a[i]*x[i])*temp;
        }
    x[n-1] = (x[n-1]-a[n-2]*x[n-2])/(b[n-1]-a[n-2]*c[n-2]);
    for (i=n-2;i>-1;i--){
        x[i] = x[i]-c[i]*x[i+1];
        }
    }
