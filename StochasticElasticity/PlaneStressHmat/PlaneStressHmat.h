void forward(double *h, const double *e_, const double *nu_, int N){
  for(int i=0;i<N;i++){
    double nu = nu_[i], E = e_[i];
    h[9*i+0] = E/(1-nu*nu);
    h[9*i+1] = E*nu/(1-nu*nu);
    h[9*i+3] = E*nu/(1-nu*nu);
    h[9*i+4] = E/(1-nu*nu);
    h[9*i+8] = E/(2.0*(1+nu));
    h[9*i+2] = 0.0;
    h[9*i+5] = 0.0;
    h[9*i+6] = 0.0;
    h[9*i+7] = 0.0;
  }
}

void backward(double *grad_e, double *grad_nu,
    const double *grad_h, 
    const double *h, const double *e_, const double *nu_, int N){
    for(int i=0;i<N;i++){
      double nu = nu_[i], E = e_[i];
      grad_e[i] = 0.0;
      grad_nu[i] = 0.0;


      grad_e[i] += grad_h[9*i+0] * (1.0/(1-nu*nu));
      grad_e[i] += grad_h[9*i+1] * (1.0*nu/(1-nu*nu));
      grad_e[i] += grad_h[9*i+3] * (1.0*nu/(1-nu*nu));
      grad_e[i] += grad_h[9*i+4] * (1.0/(1-nu*nu));
      grad_e[i] += grad_h[9*i+8] * (1.0/(2.0*(1+nu)));

      grad_nu[i] += grad_h[9*i+0] * (2*nu*E/(1-nu*nu)/(1-nu*nu));
      grad_nu[i] += grad_h[9*i+1] * (E/(1-nu*nu) + 2*nu*nu*E/((1-nu*nu)*(1-nu*nu)));
      grad_nu[i] += grad_h[9*i+3] * (E/(1-nu*nu) + 2*nu*nu*E/((1-nu*nu)*(1-nu*nu)));
      grad_nu[i] += grad_h[9*i+4] * (2*nu*E/(1-nu*nu)/(1-nu*nu));
      grad_nu[i] += grad_h[9*i+8] * (-E/(2.0*(1+nu)*(1+nu)));
    }
}