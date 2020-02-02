#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>


#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif
using namespace tensorflow;
#include "PlaneStressHmat.h"


REGISTER_OP("PlaneStressHmat")

.Input("e : double")
.Input("nu : double")
.Output("h : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle e_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &e_shape));
        shape_inference::ShapeHandle nu_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &nu_shape));

        c->set_output(0, c->MakeShape({-1,3,3}));
    return Status::OK();
  });

REGISTER_OP("PlaneStressHmatGrad")

.Input("grad_h : double")
.Input("h : double")
.Input("e : double")
.Input("nu : double")
.Output("grad_e : double")
.Output("grad_nu : double");


class PlaneStressHmatOp : public OpKernel {
private:
  
public:
  explicit PlaneStressHmatOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& e = context->input(0);
    const Tensor& nu = context->input(1);
    
    
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    
    
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);

    // extra check
        
    // create output shape
    int N = e_shape.dim_size(0);
    TensorShape h_shape({N,3,3});
            
    // create output tensor
    
    Tensor* h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, h_shape, &h));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto h_tensor = h->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(h_tensor, e_tensor, nu_tensor, N);

  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStressHmat").Device(DEVICE_CPU), PlaneStressHmatOp);



class PlaneStressHmatGradOp : public OpKernel {
private:
  
public:
  explicit PlaneStressHmatGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_h = context->input(0);
    const Tensor& h = context->input(1);
    const Tensor& e = context->input(2);
    const Tensor& nu = context->input(3);
    
    
    const TensorShape& grad_h_shape = grad_h.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    
    
    DCHECK_EQ(grad_h_shape.dims(), 3);
    DCHECK_EQ(h_shape.dims(), 3);
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int N = e_shape.dim_size(0);
    TensorShape grad_e_shape(e_shape);
    TensorShape grad_nu_shape(nu_shape);
            
    // create output tensor
    
    Tensor* grad_e = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_e_shape, &grad_e));
    Tensor* grad_nu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_nu_shape, &grad_nu));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto grad_h_tensor = grad_h.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_e_tensor = grad_e->flat<double>().data();
    auto grad_nu_tensor = grad_nu->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_e_tensor, grad_nu_tensor, grad_h_tensor, h_tensor, e_tensor, nu_tensor, N);
  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStressHmatGrad").Device(DEVICE_CPU), PlaneStressHmatGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class PlaneStressHmatOpGPU : public OpKernel {
private:
  
public:
  explicit PlaneStressHmatOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(2, context->num_inputs());
    
    
    const Tensor& e = context->input(0);
    const Tensor& nu = context->input(1);
    
    
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    
    
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    TensorShape h_shape({-1,3,3});
            
    // create output tensor
    
    Tensor* h = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, h_shape, &h));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto h_tensor = h->flat<double>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStressHmat").Device(DEVICE_GPU), PlaneStressHmatOpGPU);

class PlaneStressHmatGradOpGPU : public OpKernel {
private:
  
public:
  explicit PlaneStressHmatGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_h = context->input(0);
    const Tensor& h = context->input(1);
    const Tensor& e = context->input(2);
    const Tensor& nu = context->input(3);
    
    
    const TensorShape& grad_h_shape = grad_h.shape();
    const TensorShape& h_shape = h.shape();
    const TensorShape& e_shape = e.shape();
    const TensorShape& nu_shape = nu.shape();
    
    
    DCHECK_EQ(grad_h_shape.dims(), 3);
    DCHECK_EQ(h_shape.dims(), 3);
    DCHECK_EQ(e_shape.dims(), 1);
    DCHECK_EQ(nu_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_e_shape(e_shape);
    TensorShape grad_nu_shape(nu_shape);
            
    // create output tensor
    
    Tensor* grad_e = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_e_shape, &grad_e));
    Tensor* grad_nu = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_nu_shape, &grad_nu));
    
    // get the corresponding Eigen tensors for data access
    
    auto e_tensor = e.flat<double>().data();
    auto nu_tensor = nu.flat<double>().data();
    auto grad_h_tensor = grad_h.flat<double>().data();
    auto h_tensor = h.flat<double>().data();
    auto grad_e_tensor = grad_e->flat<double>().data();
    auto grad_nu_tensor = grad_nu->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("PlaneStressHmatGrad").Device(DEVICE_GPU), PlaneStressHmatGradOpGPU);

#endif