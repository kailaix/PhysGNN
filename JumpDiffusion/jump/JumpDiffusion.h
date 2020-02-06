#include <set>
#include <vector>
#include <algorithm>    // std::set_union, std::sort
#include <tuple>        // std::tuple, std::get, std::tie, std::ignore

using std::set;
using std::vector;

void forward(double *S, const double *y, const double *tau, double T, int N, 
    const double*dw, const double *a, const double *b, const double *bp,
    const double *c, int m){
    set<double> JumpSet(tau, tau+m);
    vector<double> ts;
    for(int i=0;i<N+1;i++) ts.push_back(i*T/double(N));
    for(int i=0;i<m;i++) ts.push_back(tau[i]);
    vector<std::tuple<double, double, double, double, double>> vect;
    for(int i=0;i<m+N+1;i++){
      vect.push_back(std::make_tuple(
        ts[i], a[i], b[i], bp[i], c[i]
      ));
    }
    std::sort(vect.begin(), vect.end());
    std::sort(ts.begin(), ts.end());

    double s = 0.0;
    int ky = 0, ks = 1;
    for(int i=1;i<m+N+1;i++){
        double h = ts[i]-ts[i-1];
        auto a_ = std::get<1>(vect[i]);
        auto b_ = std::get<2>(vect[i]);
        auto bp_ = std::get<3>(vect[i]);
        auto c_ = std::get<4>(vect[i]);
        s += a_ * h + b_ * dw[i-1] * sqrt(h) + 0.5 * bp_ * b_ * (dw[i-1]*dw[i-1]*h - h);
        if (JumpSet.count(ts[i])){
          s += c_*(y[ky++]-1);
        }else{
          S[ks++] = s;
        }
    }
}


void backward(double *grad_a, double *grad_b, double *grad_bp, double *grad_y, const double *grad_S, const double *S, const double *y, const double *tau, double T, int N, 
    const double*dw, const double *a, const double *b, const double *bp,
    const double *c, int m){
    set<double> JumpSet(tau, tau+m);
    vector<double> ts;
    for(int i=0;i<N+1;i++) ts.push_back(i*T/double(N));
    for(int i=0;i<m;i++) ts.push_back(tau[i]);
    vector<std::tuple<double, double, double, double, double, int>> vect;
    for(int i=0;i<m+N+1;i++){
      vect.push_back(std::make_tuple(
        ts[i], a[i], b[i], bp[i], c[i], i
      ));
    }
    std::sort(vect.begin(), vect.end());
    std::sort(ts.begin(), ts.end());

    double s = 0.0;
    int ky = m-1, ks = N;

    for(int i=m+N;i>0;i--){        
      // printf("%d\n", i);
     
        if (JumpSet.count(ts[i])){
          auto c_ = std::get<4>(vect[i]);
          grad_y[ky--] = c_*s;
        }else{
          s += grad_S[ks--];
        }

        double h = ts[i]-ts[i-1];
        auto b_ = std::get<2>(vect[i]);
        auto bp_ = std::get<3>(vect[i]);
        int idx = std::get<5>(vect[i]);
        grad_a[idx] = s * h;
        grad_b[idx] = s * (dw[i-1] * sqrt(h) + 0.5 * bp_ * (dw[i-1]*dw[i-1]*h - h));
        grad_bp[idx] = s *  0.5 * b_ * (dw[i-1]*dw[i-1]*h - h);

    }
}


// 2D pure jump diffusion 
void forward2(double *S, const double *y, const double *tau, double T, int N, 
    const double *c, int n, int m){
      // printf("T = %f, N = %d, n = %d, m = %d\n", T, N, n, m);
    auto y1 = y, y2 = y + n;
    auto S1 = S, S2 = S + N + 1;
    for(int i=0;i<2*N+2;i++) S[i] = 0.0;

    set<double> JumpSet(tau, tau+m);
    vector<double> ts;
    for(int i=0;i<N+1;i++) ts.push_back(i*T/double(N));
    for(int i=0;i<m;i++) ts.push_back(tau[i]);
    vector<std::tuple<double, double>> vect;
    for(int i=0;i<m+N+1;i++){
      vect.push_back(std::make_tuple(
        ts[i], c[i]
      ));
    }
    std::sort(vect.begin(), vect.end());
    std::sort(ts.begin(), ts.end());

    double s1 = 0.0;
    double s2 = 0.0;
    int ky = 0, ks = 1;
    for(int i=1;i<m+N+1;i++){
      // printf("i=%d\n", i);
        double h = ts[i]-ts[i-1];
        auto c_ = std::get<1>(vect[i]);
        if (JumpSet.count(ts[i])){
          s1 += c_*(y1[ky]-1);
          s2 += c_*(y2[ky]-1);
          ky += 1;
        }else{
          S1[ks] = s1;
          S2[ks] = s2;
          ks += 1;
        }
    }
}



void backward2(double *grad_y, const double * grad_S, 
    const double *S, const double *y, const double *tau, double T, int N, 
    const double *c, int n, int m){
      // printf("T = %f, N = %d, n = %d, m = %d\n", T, N, n, m);

    set<double> JumpSet(tau, tau+m);
    vector<double> ts;
    for(int i=0;i<N+1;i++) ts.push_back(i*T/double(N));
    for(int i=0;i<m;i++) ts.push_back(tau[i]);
    vector<std::tuple<double, double>> vect;
    for(int i=0;i<m+N+1;i++){
      vect.push_back(std::make_tuple(
        ts[i], c[i]
      ));
    }
    std::sort(vect.begin(), vect.end());
    std::sort(ts.begin(), ts.end());

    double s1 = 0.0;
    double s2 = 0.0;
    int ky = m-1, ks = N;
    auto grad_y1 = grad_y, grad_y2 = grad_y + n;
    auto grad_S1 = grad_S, grad_S2 = grad_S + N + 1;
    for(int i=0;i<2*n;i++) grad_y[i] = 0.0;

    for(int i=m+N;i>0;i--){
      // printf("%d: %f %f\n", i, s1, s2);
        if (JumpSet.count(ts[i])){
          auto c_ = std::get<1>(vect[i]);
          grad_y1[ky] = c_*s1;
          grad_y2[ky] = c_*s2;
          ky -= 1;
        }else{
          s1 += grad_S1[ks];
          s2 += grad_S2[ks];
          ks -= 1;
        }
    }
}