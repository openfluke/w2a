#include "welvet/architecture.hpp"
#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_chain.hpp"
#include "welvet/runtime/tween_chain_train.hpp"
#include "welvet/metal/device.hpp"
#include "welvet/metal/training.hpp"
#include "welvet/runtime/training.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

using namespace welvet;
void check(bool ok, const char* msg) {
  if (!ok) { std::fprintf(stderr, "%s\n", msg); std::exit(1); }
}
void success(Status s) { check(s.ok(), s.message.c_str()); }
void blocked(Status s) {
  if (s.code != StatusCode::Unsupported || s.message.find("CPU execution is forbidden") == std::string::npos) std::fprintf(stderr, "unexpected status=%d: %s\n", int(s.code), s.message.c_str());
  check(s.code == StatusCode::Unsupported && s.message.find("CPU execution is forbidden") != std::string::npos,
        "CPU operation did not fail closed");
}
int main() {
  TensorF32 in, pre, post, grad;
  // Unported kinds and legacy host math must remain unavailable.
  { kmeans::Layer l; blocked(kmeans::forward(l, in, &pre, &post)); }
  { parallel::Layer l; blocked(parallel::forward(l, in, &pre, &post)); }
  WeightStore w;
  float z = 0;
  blocked(w.matvec_f32(&z, &z)); blocked(w.matvect_f32(&z, &z));
  float init[] = {1, -2};
  double dg[] = {0.25, -0.5};
  for (int dt = 0; dt < kNumDTypes; ++dt) {
    if (dt == int(DType::Float32)) continue;
    success(WeightStore::make(1, 2, DType(dt), init, &w));
    success(w.apply_sgd(dg, 0.1));
    check(w.dtype==DType(dt) && !w.native.empty() && w.f32.empty(), "SGD replaced native storage with an f32 master");
  }
  success(WeightStore::make_f32(1, 2, init, &w));
  w.bias = {1.0}; success(w.apply_bias_sgd(dg, 2, 0.1));
  check(std::abs(w.bias[0]-0.975)<1e-12, "GPU bias SGD mismatch");

  // No-device failures must be errors, not successful host execution.
  if (!metal::available()) {
    float p[] = {1}, t[] = {0}, g[] = {1}; double loss;
    check(!metal::mse_f32(p, t, 1, &loss, g), "MSE fell back without Metal");
    check(!metal::sgd_f32(p, g, 1, 0.1), "SGD fell back without Metal");
    check(!metal::accumulate_f32(p, g, 1), "accumulate fell back without Metal");
    std::puts("No-device rejection passed; GPU validation requires device access");
    return 2;
  }
  // CPU references belong exclusively in w2a. Use a non-power-of-two count and
  // cancellation-sensitive data to exercise the float2 loss/gradient arithmetic.
  float p[] = {1.0001f, -3.25f, 7.5f, 0.00001f, -0.75f, 4, -2};
  float t[] = {1, -2.75f, -1.5f, -0.00002f, 0.1f, 2, 3};
  float g[7]; double loss;
  success(metal::mse_f32(p, t, 7, &loss, g));
  double expected = 0;
  for (int i = 0; i < 7; ++i) {
    double d = double(p[i])-double(t[i]); expected += d*d;
    check(g[i] == float((2.0/7.0)*d), "GPU MSE gradient mismatch");
  }
  expected /= 7;
  check(std::abs(loss-expected) <= 1e-12 * std::max(1.0, std::abs(expected)), "GPU loss outside declared band");
  float weights[7], want[7]; const double lr = 0.1234567890123;
  for (int i = 0; i < 7; ++i) weights[i] = want[i] = p[i];
  for (int step = 0; step < 3; ++step) {
    success(metal::sgd_f32(weights, g, 7, lr));
    for (int i = 0; i < 7; ++i) {
      want[i] -= float(lr * double(g[i]));
      check(weights[i] == want[i], "GPU SGD mismatch");
    }
  }
  success(metal::accumulate_f32(weights, p, 7));
  for (int i = 0; i < 7; ++i) check(weights[i] == want[i] + p[i], "GPU accumulation mismatch");
  success(w.apply_sgd(dg, lr));
  check(w.f32[0] == init[0]-float(lr*dg[0]) && w.f32[1] == init[1]-float(lr*dg[1]), "GPU double gradient SGD mismatch");
  check(!metal::sgd_f32(weights, g, 7, std::numeric_limits<double>::quiet_NaN()), "NaN LR accepted");

  // Exercise every Dense activation independently of the checklist's Linear/ReLU fixtures.
  for(int act=-1;act<=6;++act) {
    dense::Layer activation_layer;float one[]={1};
    success(dense::new_layer(1,1,Activation(act),DType::Float32,one,&activation_layer));
    TensorF32 x;x.shape={5,1};x.data={-3,-0.5f,0.001f,0.5f,3};TensorF32 pp,yy;
    success(dense::forward_f32(activation_layer,x,&pp,&yy));
    TensorF32 dy;dy.shape=x.shape;dy.data.assign(5,1);TensorF32 dx,dw;
    success(dense::backward_f32(activation_layer,dy,x,pp,&dx,&dw));
    for(int i=0;i<5;++i) {
      double v=x.data[i], y=v, derivative=1, sig=1/(1+std::exp(-v));
      if(act==0){y=std::max(v,0.0);derivative=v>0?1:0;}
      if(act==1){y=v*sig;derivative=sig*(1+v*(1-sig));}
      if(act==2){double k=0.7978845608028654,u=k*(v+0.044715*v*v*v),tt=std::tanh(u);y=0.5*v*(1+tt);derivative=0.5*(1+tt)+0.5*v*(1-tt*tt)*k*(1+3*0.044715*v*v);}
      if(act==3){y=std::tanh(v);derivative=1-y*y;}
      if(act==4){y=sig;derivative=sig*(1-sig);}
      if(act==5){y=v<0?0.01*v:v;derivative=v>0?1:0.01;}
      if(act==6){y=v<0?0:v*v;derivative=v>0?2*v:0;}
      check(std::abs(yy.data[i]-float(y))<2e-7,"Dense activation mismatch");
      check(std::abs(dx.data[i]-float(derivative))<2e-7,"Dense activation derivative mismatch");
    }
  }
  // Projection must use extended sums and preserve Go block-average mapping.
  TensorF32 gap, projected;
  gap.shape = {4}; gap.data = {2,4,6,8};
  success(systems::tween::project_gap(gap, {2}, &projected));
  check(projected.data == std::vector<float>({3,7}), "Tween downsample mismatch");
  success(systems::tween::project_gap(gap, {2,3}, &projected));
  check(projected.shape == std::vector<int64_t>({2,3}) &&
        projected.data == std::vector<float>({2,2,4,6,6,8}), "Tween upsample mismatch");
  gap.shape = {3}; gap.data = {1e8f,1,-1e8f};
  success(systems::tween::project_gap(gap, {1}, &projected));
  check(projected.data[0] == float(1.0/3.0), "Tween projection lost cancellation residual");
  check(!systems::tween::project_gap(gap, {0}, &projected), "Tween zero shape accepted");
  check(!systems::tween::project_gap(gap, {-1}, &projected), "Tween negative shape accepted");
  check(!systems::tween::project_gap(gap, {INT64_MAX,2}, &projected), "Tween oversized shape accepted");
  check(!systems::tween::project_gap(gap, {1}, nullptr), "Tween nil projection accepted");
  // Nested projected credit feeds the original tape to every local update.
  stack::Child nested;
  nested.kind = stack::Kind::NestedStack;
  nested.nested.resize(2);
  for (int i=0; i<2; ++i) {
    nested.nested[i].kind = stack::Kind::Dense;
    float initial[] = {float(i+1)};
    success(dense::new_layer(1,1,Activation::Linear,DType::Float32,initial,
                            &nested.nested[i].dense_op));
  }
  TensorF32 nx, ng;
  nx.shape={1,1}; nx.data={1}; ng=nx;
  success(systems::tween::apply_projected(nested, ng, nx, 0.1));
  check(nested.nested[0].dense_op.weights.f32[0] == 0.9f &&
        nested.nested[1].dense_op.weights.f32[0] == 1.9f,
        "Nested Tween used chained or updated-input credit");
  // True chain-rule credit uses old downstream weights and the full rate.
  stack::Layer chain;
  chain.children.resize(2);
  for (int i=0; i<2; ++i) {
    chain.children[i].kind=stack::Kind::Dense;
    float initial[]={float(i+1)};
    success(dense::new_layer(1,1,Activation::Linear,DType::Float32,initial,
                            &chain.children[i].dense_op));
  }
  TensorF32 cx, ct;
  cx.shape={1,1}; cx.data={1}; ct=cx; ct.data={0};
  double chain_loss=0;
  success(runtime::tween_chain_train::train_stack_mse(chain,cx,ct,0.1,&chain_loss));
  check(chain_loss==4 && chain.children[0].dense_op.weights.f32[0]==1.0f-0.8f &&
        chain.children[1].dense_op.weights.f32[0]==2.0f-0.4f,
        "TweenChain leaf lost chain-rule/full-rate credit");
  auto chain_grid=architecture::new_grid(1,1,2,1);
  for (int i=0; i<2; ++i) {
    dense::Layer leaf; float initial[]={float(i+1)};
    success(dense::new_layer(1,1,Activation::Linear,DType::Float32,initial,&leaf));
    success(architecture::bind_dense(chain_grid,0,0,i,0,std::move(leaf)));
  }
  runtime::forward::Result chain_fwd;
  runtime::backward::Result chain_bwd;
  success(runtime::forward::volume_dense(chain_grid,cx,&chain_fwd));
  success(systems::tween_chain::train_volume_mse(chain_grid,chain_fwd,ct,0.1,&chain_loss,&chain_bwd));
  check(chain_loss==4 && chain_bwd.grad_in.data==std::vector<float>({8}) &&
        chain_grid.cells[0].dense_op.weights.f32[0]==1.0f-0.8f &&
        chain_grid.cells[1].dense_op.weights.f32[0]==2.0f-0.4f,
        "TweenChain volume lost true input gradient/full-rate credit");
  check(!runtime::tween_chain_train::train_stack_mse(chain,cx,ct,0.1,nullptr),
        "TweenChain nil loss accepted");
  stack::Layer empty_chain;
  check(!runtime::tween_chain_train::train_stack_mse(empty_chain,cx,ct,0.1,&chain_loss),
        "TweenChain empty stack accepted");
  // GPU cache must observe each new weight update, including changed batch size.
  dense::Layer l;
  float wi[] = {1}; success(dense::new_layer(1, 1, Activation::Linear, DType::Float32, wi, &l));
  in.shape = {1,1}; in.data = {2}; grad.shape = {1,1}; grad.data = {0.5f};
  for (int i = 0; i < 3; ++i) {
    success(dense::forward_f32(l, in, &pre, &post));
    check(post.data[0] == 2.0f*(1.0f-0.125f*i), "stale resident GPU weights");
    success(dense::apply_grad_sgd(&l, grad, 0.25));
  }
  in.shape = {2,1}; in.data = {2,4};
  success(dense::forward_f32(l, in, &pre, &post));
  check(post.data[0] == 1.25f && post.data[1] == 2.5f, "batch remount lost SGD weights");
  in.data.clear(); check(!dense::forward_f32(l, in, &pre, &post), "short Dense input accepted");
  std::puts("GPU loss, gradients, SGD, mesh accumulation, cache coherence and CPU rejection passed");
}
