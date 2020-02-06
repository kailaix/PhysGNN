#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include "JumpDiffusion.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("JumpDiffusion")
.Input("y : double")
.Input("tau : double")
.Input("t : double")
.Input("n : int32")
.Input("m : int32")
.Input("a : double")
.Input("b : double")
.Input("bp : double")
.Input("c : double")
.Input("dw : double")
.Output("s : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &y_shape));
        shape_inference::ShapeHandle tau_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &tau_shape));
        shape_inference::ShapeHandle t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &t_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &n_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &m_shape));
        shape_inference::ShapeHandle a_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &a_shape));
        shape_inference::ShapeHandle b_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &b_shape));
        shape_inference::ShapeHandle bp_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &bp_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 1, &c_shape));
        shape_inference::ShapeHandle dw_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 1, &dw_shape));

        c->set_output(0, c->Vector(-1));
    return Status::OK();
  });

REGISTER_OP("JumpDiffusionGrad")
.Input("grad_s : double")
.Input("s : double")
.Input("y : double")
.Input("tau : double")
.Input("t : double")
.Input("n : int32")
.Input("m : int32")
.Input("a : double")
.Input("b : double")
.Input("bp : double")
.Input("c : double")
.Input("dw : double")
.Output("grad_y : double")
.Output("grad_tau : double")
.Output("grad_t : double")
.Output("grad_n : int32")
.Output("grad_m : int32")
.Output("grad_a : double")
.Output("grad_b : double")
.Output("grad_bp : double")
.Output("grad_c : double")
.Output("grad_dw : double");


class JumpDiffusionOp : public OpKernel {
private:
  
public:
  explicit JumpDiffusionOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(10, context->num_inputs());
    
    
    const Tensor& y = context->input(0);
    const Tensor& tau = context->input(1);
    const Tensor& t = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& a = context->input(5);
    const Tensor& b = context->input(6);
    const Tensor& bp = context->input(7);
    const Tensor& c = context->input(8);
    const Tensor& dw = context->input(9);
    
    
    const TensorShape& y_shape = y.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& bp_shape = bp.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& dw_shape = dw.shape();
    
    
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(bp_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(dw_shape.dims(), 1);

    // extra check
        
    // create output shape
    
    int m_ = *m.flat<int32>().data();
    int N = *n.flat<int32>().data();
    TensorShape s_shape({N+1});
    DCHECK_GE(tau_shape.dim_size(0), m_);
    DCHECK_GE(y_shape.dim_size(0), m_);
    DCHECK_GE(a_shape.dim_size(0), m_+N+1);
    DCHECK_GE(b_shape.dim_size(0), m_+N+1);
    DCHECK_GE(bp_shape.dim_size(0), m_+N+1);
    DCHECK_GE(c_shape.dim_size(0), m_+N+1);
    DCHECK_GE(dw_shape.dim_size(0), m_+N+1);
            
    // create output tensor
    
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, s_shape, &s));
    
    // get the corresponding Eigen tensors for data access
    
    auto y_tensor = y.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto n_tensor = n.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto bp_tensor = bp.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto dw_tensor = dw.flat<double>().data();
    auto s_tensor = s->flat<double>().data();   

    // implement your forward function here 
    for(int i=0;i<N+1;i++) s_tensor[i] = 0.0;
    // TODO:
    forward(s_tensor, y_tensor, tau_tensor, *t_tensor, N, 
        dw_tensor, a_tensor, b_tensor, bp_tensor, c_tensor, m_);
    

  }
};
REGISTER_KERNEL_BUILDER(Name("JumpDiffusion").Device(DEVICE_CPU), JumpDiffusionOp);



class JumpDiffusionGradOp : public OpKernel {
private:
  
public:
  explicit JumpDiffusionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_s = context->input(0);
    const Tensor& s = context->input(1);
    const Tensor& y = context->input(2);
    const Tensor& tau = context->input(3);
    const Tensor& t = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& a = context->input(7);
    const Tensor& b = context->input(8);
    const Tensor& bp = context->input(9);
    const Tensor& c = context->input(10);
    const Tensor& dw = context->input(11);
    
    
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& a_shape = a.shape();
    const TensorShape& b_shape = b.shape();
    const TensorShape& bp_shape = bp.shape();
    const TensorShape& c_shape = c.shape();
    const TensorShape& dw_shape = dw.shape();
    
    
    DCHECK_EQ(grad_s_shape.dims(), 1);
    DCHECK_EQ(s_shape.dims(), 1);
    DCHECK_EQ(y_shape.dims(), 1);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(a_shape.dims(), 1);
    DCHECK_EQ(b_shape.dims(), 1);
    DCHECK_EQ(bp_shape.dims(), 1);
    DCHECK_EQ(c_shape.dims(), 1);
    DCHECK_EQ(dw_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_y_shape(y_shape);
    TensorShape grad_tau_shape(tau_shape);
    TensorShape grad_t_shape(t_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_a_shape(a_shape);
    TensorShape grad_b_shape(b_shape);
    TensorShape grad_bp_shape(bp_shape);
    TensorShape grad_c_shape(c_shape);
    TensorShape grad_dw_shape(dw_shape);
            
    // create output tensor
    int m_ = *m.flat<int32>().data();
    int N = *n.flat<int32>().data();
    
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_y_shape, &grad_y));
    Tensor* grad_tau = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_tau_shape, &grad_tau));
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_t_shape, &grad_t));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_m_shape, &grad_m));
    Tensor* grad_a = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_a_shape, &grad_a));
    Tensor* grad_b = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(6, grad_b_shape, &grad_b));
    Tensor* grad_bp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(7, grad_bp_shape, &grad_bp));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(8, grad_c_shape, &grad_c));
    Tensor* grad_dw = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(9, grad_dw_shape, &grad_dw));
    
    // get the corresponding Eigen tensors for data access
    
    auto y_tensor = y.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto n_tensor = n.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto a_tensor = a.flat<double>().data();
    auto b_tensor = b.flat<double>().data();
    auto bp_tensor = bp.flat<double>().data();
    auto c_tensor = c.flat<double>().data();
    auto dw_tensor = dw.flat<double>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();
    auto grad_tau_tensor = grad_tau->flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_a_tensor = grad_a->flat<double>().data();
    auto grad_b_tensor = grad_b->flat<double>().data();
    auto grad_bp_tensor = grad_bp->flat<double>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();
    auto grad_dw_tensor = grad_dw->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    for(int i=0;i<grad_y_shape.dim_size(0);i++) grad_y_tensor[i] = 0.0;
    for(int i=0;i<grad_a_shape.dim_size(0);i++) grad_a_tensor[i] = 0.0;
    for(int i=0;i<grad_b_shape.dim_size(0);i++) grad_b_tensor[i] = 0.0;
    for(int i=0;i<grad_bp_shape.dim_size(0);i++) grad_bp_tensor[i] = 0.0;
    backward(grad_a_tensor, grad_b_tensor, grad_bp_tensor,  grad_y_tensor, grad_s_tensor, s_tensor, y_tensor, tau_tensor, *t_tensor, N, 
        dw_tensor, a_tensor, b_tensor, bp_tensor, c_tensor, m_);
    
    
  }
};
REGISTER_KERNEL_BUILDER(Name("JumpDiffusionGrad").Device(DEVICE_CPU), JumpDiffusionGradOp);


REGISTER_OP("MultivariateJumpDiffusion")
.Input("y : double")
.Input("tau : double")
.Input("t : double")
.Input("n : int32")
.Input("m : int32")
.Input("c : double")
.Output("s : double")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle y_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &y_shape));
        shape_inference::ShapeHandle tau_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &tau_shape));
        shape_inference::ShapeHandle t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &t_shape));
        shape_inference::ShapeHandle n_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &n_shape));
        shape_inference::ShapeHandle m_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &m_shape));
        shape_inference::ShapeHandle c_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &c_shape));

        c->set_output(0, c->Matrix(2,-1));
    return Status::OK();
  });

REGISTER_OP("MultivariateJumpDiffusionGrad")

.Input("grad_s : double")
  .Input("s : double")
  .Input("y : double")
  .Input("tau : double")
  .Input("t : double")
  .Input("n : int32")
  .Input("m : int32")
  .Input("c : double")
  .Output("grad_y : double")
  .Output("grad_tau : double")
  .Output("grad_t : double")
  .Output("grad_n : int32")
  .Output("grad_m : int32")
  .Output("grad_c : double");


class MultivariateJumpDiffusionOp : public OpKernel {
private:
  
public:
  explicit MultivariateJumpDiffusionOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(6, context->num_inputs());
    
    
    const Tensor& y = context->input(0);
    const Tensor& tau = context->input(1);
    const Tensor& t = context->input(2);
    const Tensor& n = context->input(3);
    const Tensor& m = context->input(4);
    const Tensor& c = context->input(5);
    
    
    const TensorShape& y_shape = y.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& c_shape = c.shape();
    
    
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(c_shape.dims(), 1);
    

    // extra check
        
    // create output shape
    
    int N = *n.flat<int32>().data();
    int m_ = *m.flat<int32>().data();
    int n_ = y_shape.dim_size(1);
    TensorShape s_shape({2,N+1});



    DCHECK_EQ(n_, m_);
    DCHECK_EQ(c_shape.dim_size(0), n_+N+1);
            
    // create output tensor
    
    Tensor* s = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, s_shape, &s));
    
    // get the corresponding Eigen tensors for data access
    
    auto y_tensor = y.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto n_tensor = n.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto c_tensor = c.flat<double>().data();
    auto s_tensor = s->flat<double>().data();   

    // implement your forward function here 

    // TODO:
    forward2(s_tensor, y_tensor, tau_tensor, *t_tensor, *n_tensor, c_tensor, n_, m_);
  }
};
REGISTER_KERNEL_BUILDER(Name("MultivariateJumpDiffusion").Device(DEVICE_CPU), MultivariateJumpDiffusionOp);



class MultivariateJumpDiffusionGradOp : public OpKernel {
private:
  
public:
  explicit MultivariateJumpDiffusionGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_s = context->input(0);
    const Tensor& s = context->input(1);
    const Tensor& y = context->input(2);
    const Tensor& tau = context->input(3);
    const Tensor& t = context->input(4);
    const Tensor& n = context->input(5);
    const Tensor& m = context->input(6);
    const Tensor& c = context->input(7);
    
    
    const TensorShape& grad_s_shape = grad_s.shape();
    const TensorShape& s_shape = s.shape();
    const TensorShape& y_shape = y.shape();
    const TensorShape& tau_shape = tau.shape();
    const TensorShape& t_shape = t.shape();
    const TensorShape& n_shape = n.shape();
    const TensorShape& m_shape = m.shape();
    const TensorShape& c_shape = c.shape();
    
    
    DCHECK_EQ(grad_s_shape.dims(), 2);
    DCHECK_EQ(s_shape.dims(), 2);
    DCHECK_EQ(y_shape.dims(), 2);
    DCHECK_EQ(tau_shape.dims(), 1);
    DCHECK_EQ(t_shape.dims(), 0);
    DCHECK_EQ(n_shape.dims(), 0);
    DCHECK_EQ(m_shape.dims(), 0);
    DCHECK_EQ(c_shape.dims(), 1);

    // extra check
    // int m = Example.dim_size(0);
    int N = *n.flat<int32>().data();
    int m_ = *m.flat<int32>().data();
    int n_ = y_shape.dim_size(1);
        
    // create output shape
    
    TensorShape grad_y_shape(y_shape);
    TensorShape grad_tau_shape(tau_shape);
    TensorShape grad_t_shape(t_shape);
    TensorShape grad_n_shape(n_shape);
    TensorShape grad_m_shape(m_shape);
    TensorShape grad_c_shape(c_shape);
            
    // create output tensor
    
    Tensor* grad_y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_y_shape, &grad_y));
    Tensor* grad_tau = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_tau_shape, &grad_tau));
    Tensor* grad_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_t_shape, &grad_t));
    Tensor* grad_n = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_n_shape, &grad_n));
    Tensor* grad_m = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(4, grad_m_shape, &grad_m));
    Tensor* grad_c = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(5, grad_c_shape, &grad_c));
    
    // get the corresponding Eigen tensors for data access
    
    auto y_tensor = y.flat<double>().data();
    auto tau_tensor = tau.flat<double>().data();
    auto t_tensor = t.flat<double>().data();
    auto n_tensor = n.flat<int32>().data();
    auto m_tensor = m.flat<int32>().data();
    auto c_tensor = c.flat<double>().data();
    auto grad_s_tensor = grad_s.flat<double>().data();
    auto s_tensor = s.flat<double>().data();
    auto grad_y_tensor = grad_y->flat<double>().data();
    auto grad_tau_tensor = grad_tau->flat<double>().data();
    auto grad_t_tensor = grad_t->flat<double>().data();
    auto grad_n_tensor = grad_n->flat<int32>().data();
    auto grad_m_tensor = grad_m->flat<int32>().data();
    auto grad_c_tensor = grad_c->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward2(grad_y_tensor, grad_s_tensor, s_tensor, y_tensor, tau_tensor, *t_tensor, *n_tensor, c_tensor, n_, m_);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("MultivariateJumpDiffusionGrad").Device(DEVICE_CPU), MultivariateJumpDiffusionGradOp);
