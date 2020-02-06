#include<iostream>
void solve(double* a, double* b, double* c, double* d, int n) {
    n--; // since we start from x0 (not x1)
    c[0] /= b[0];
    d[0] /= b[0];

    for (int i = 1; i < n; i++) {
        c[i] /= b[i] - a[i]*c[i-1];
        d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);
    }

    d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

    for (int i = n; i-- > 0;) {
        d[i] -= c[i]*d[i+1];
    }
}

void forward(double *out, const double *A, const double *B, const double *C, const double *f, int n){
    double *a = new double[n];
    double *b = new double[n];
    double *c = new double[n];
    memcpy(a, A, sizeof(double)*n);
    memcpy(b+1, B, sizeof(double)*(n-1));
    memcpy(c, C, sizeof(double)*(n-1));
    memcpy(out, f, sizeof(double)*n);
    b[0] = 0.0;
    c[n-1] = 0.0;
    solve(b, a, c, out, n);
    delete [] a;
    delete [] b;
    delete [] c;
}

void backward(const double *grad_out, const double *out, const double *A, const double *B, const double *C, const double *f, int n,
                double *grad_A, double *grad_B, double *grad_C, double *grad_f){
    double *grad = new double[n];
    forward(grad, A, C, B, grad_out, n);
    memcpy(grad_f, grad, sizeof(double)*n);
    for(int i=0;i<n;i++){
        grad_A[i] = 0;
        if(i<n-1) grad_B[i] = 0;
        if(i<n-1) grad_C[i] = 0;
    }
    for(int i=0;i<n;i++){
        if(i+1<n) grad_C[i] -= out[i+1]*grad[i];
        grad_A[i] -= out[i]*grad[i];
        if(i-1>=0) grad_B[i-1] -= out[i-1]*grad[i];
    }

}
