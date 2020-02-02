#include <vector>
#include <eigen3/Eigen/Core>

using std::vector;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

class Forward{
  private:
    vector<int> ii, jj;
    vector<double> vv;
  public:
    Forward(const double *hmat, int m, int n, double h){
      Eigen::Matrix<double,8,8> Omega;
      Eigen::Matrix<double,3,8> B;
      Eigen::Matrix<double,3,3> K;
      Eigen::Vector<int,8> idx;
      // std::cout << Omega << std::endl;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){

          Omega.setZero();
          for(int i0=0;i0<3;i0++)
            for(int j0=0;j0<3;j0++)
              K(i0,j0) = hmat[9*(j*m+i)+j0*3+i0];
          for(int i0=0;i0<2;i0++)
            for(int j0=0;j0<2;j0++){
              double xi = pts[i0], eta = pts[j0];
              B << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
              Omega += B.transpose() * K * B * 0.25 * h* h;
            }

          idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1,
               j*(m+1)+i + (m+1)*(n+1), j*(m+1)+i+1 + (m+1)*(n+1), (j+1)*(m+1)+i + (m+1)*(n+1), (j+1)*(m+1)+i+1 + (m+1)*(n+1);
          for(int p=0;p<8;p++){
            for(int q=0;q<8;q++){
              ii.push_back(idx[p]+1);
              jj.push_back(idx[q]+1);
              vv.push_back(Omega(p,q));
            }
          }
        }
      }
    }

    void fill(OpKernelContext* context){
      int N = ii.size();
      TensorShape ii_shape({N});
      TensorShape jj_shape({N});
      TensorShape vv_shape({N});
              
      // create output tensor
      
      Tensor* ii_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, ii_shape, &ii_));
      Tensor* jj_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(1, jj_shape, &jj_));
      Tensor* vv_ = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(2, vv_shape, &vv_));
      
      // get the corresponding Eigen tensors for data access
      auto ii_tensor = ii_->flat<int64>().data();
      auto jj_tensor = jj_->flat<int64>().data();
      auto vv_tensor = vv_->flat<double>().data(); 
      for(int i=0;i<N;i++){
        ii_tensor[i] = ii[i];
        jj_tensor[i] = jj[i];
        vv_tensor[i] = vv[i];
      }
    }
};


void backward(
  double *grad_hmat, const double * grad_vv,  
  int m, int n, double h
){
      Eigen::Matrix<double,3,8> B;
      Eigen::Matrix<double,3,3> dK;
      Eigen::Matrix<double,8,8> dOmega;
      for(int i=0;i<9*m*n;i++) grad_hmat[i] = 0.0;
      int k = 0;
      for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
          dOmega.setZero();
          for(int p=0;p<8;p++){
            for(int q=0;q<8;q++){
              dOmega(p, q) = grad_vv[k++];
            }
          }

          dK.setZero();
          for(int i=0;i<2;i++)
            for(int j=0;j<2;j++){
              double xi = pts[i], eta = pts[j];
              B << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi,
                -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi, -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta;
              dK += B * dOmega * B.transpose() * 0.25 * h* h;
            }

          for(int i0=0;i0<3;i0++){
            for(int j0=0;j0<3;j0++){
              grad_hmat[(j*m+i)*9+j0*3+i0] += dK(i0,j0);
            }
          }
            
        }
      }
    };