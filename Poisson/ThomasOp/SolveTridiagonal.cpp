#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
using namespace tensorflow;
// #include "ADEL.h"
#include "SolveTridiagonal.h"

REGISTER_OP("SolveTridiagonal")
  
  .Input("a : double")
  .Input("b : double")
  .Input("c : double")
  .Input("f : double")
  .Output("d : double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // c->set_output(0, c->Vector(-1)); 
    return Status::OK();
  });
class SolveTridiagonalOp : public OpKernel {
private:
  
public:
  explicit SolveTridiagonalOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(4, context->num_inputs());
    
    
    const Tensor& a = context->input(0);
    const Tensor& b = context->input(1);
    const Tensor& c = context->input(2);
    const Tensor& f = context->input(3);
    
    
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    int n = a_shape.dim_size(0);
    DCHECK_EQ(b_shape.dim_size(0), n-1);
    DCHECK_EQ(c_shape.dim_size(0), n-1);
    DCHECK_EQ(f_shape.dim_size(0), n);
    TensorShape d_shape({n});
            
    // create output tensor
    
    Tensor* d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, d_shape, &d));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto d_tensor = d->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward(d_tensor, a_tensor, b_tensor, c_tensor, f_tensor, n);
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveTridiagonal").Device(DEVICE_CPU), SolveTridiagonalOp);


REGISTER_OP("SolveTridiagonalGrad")
  
  .Input("grad_d : double")
  .Input("d : double")
  .Input("a : double")
  .Input("b : double")
  .Input("c : double")
  .Input("f : double")
  .Output("grad_a : double")
  .Output("grad_b : double")
  .Output("grad_c : double")
  .Output("grad_f : double");
class SolveTridiagonalGradOp : public OpKernel {
private:
  
public:
  explicit SolveTridiagonalGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_d = context->input(0);
    const Tensor& d = context->input(1);
    const Tensor& a = context->input(2);
    const Tensor& b = context->input(3);
    const Tensor& c = context->input(4);
    const Tensor& f = context->input(5);
    
    
    const TensorShape& grad_d_shape = grad_d.shape();
    const TensorShape& d_shape = d.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& f_shape = f.shape();
    
    
    DCHECK_EQ(grad_d_shape.dims(), 1);
    DCHECK_EQ(d_shape.dims(), 1);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(f_shape.dims(), 1);

    // extra check
    int n = a_shape.dim_size(0);
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_f_shape(f_shape);
            
    // create output tensor
    
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_b_shape, &grad_b));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_c_shape, &grad_c));
    Tensor* grad_f = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_f_shape, &grad_f));
    
    // get the corresponding Eigen tensors for data access
    
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto f_tensor = f.flat<double>().data();
    auto grad_d_tensor = grad_d.flat<double>().data();
    auto d_tensor = d.flat<double>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_f_tensor = grad_f->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_d_tensor, d_tensor, a_tensor, b_tensor, c_tensor, f_tensor, 
          n, grad_a_tensor, grad_b_tensor, grad_c_tensor, grad_f_tensor);
  }
};
REGISTER_KERNEL_BUILDER(Name("SolveTridiagonalGrad").Device(DEVICE_CPU), SolveTridiagonalGradOp);
