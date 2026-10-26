#include "welvet/cam_sync.hpp"
#include "welvet/runtime/cameral_train.hpp"
#include "welvet/architecture.hpp"
#include "welvet_c.h"
#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_split.hpp"
#include "welvet/systems/tween_chain.hpp"
#include "welvet/runtime/tween_chain_train.hpp"
#include "welvet/runtime/tween_split_train.hpp"
#include "welvet/runtime/tween_alt_train.hpp"
#include "welvet/runtime/line.hpp"
#include "welvet/runtime/step.hpp"
#include <cstring>
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
  // Public FFI rejects malformed descriptors before allocating/executing them.
  check(welvet_model_abi_version()==1,"unexpected model ABI version");
  welvet_model_session* bad_session=nullptr;
  check(welvet_model_mount(nullptr,0,&bad_session).code==1 && !bad_session,"null model mount accepted");
  std::vector<uint8_t> descriptor;
  auto put_i32=[&](int32_t x){auto* p=reinterpret_cast<uint8_t*>(&x);descriptor.insert(descriptor.end(),p,p+4);};
  put_i32(0x314e4d57);put_i32(0);put_i32(1);
  for(int i=0;i<4;++i)put_i32(1);
  put_i32(7); // MHA: deliberately invalid zero heads, no payload required.
  for(int32_t x:{1,1,0,0,0,1,0,0,1})put_i32(x);
  check(welvet_model_mount(descriptor.data(),descriptor.size(),&bad_session).code==1 && !bad_session,"invalid MHA descriptor accepted");
  welvet_metal_blob empty{};welvet_metal_blob_free(&empty);welvet_metal_blob_free(nullptr);welvet_model_release(nullptr);
  TensorF32 in, pre, post, grad;
  // Legacy host numerical APIs must remain unavailable.
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
  // KMeans and Parallel now run their complete NormalBP path on Metal.
  { float weights[]={0.1f,0.3f,-0.2f,0.4f};TensorF32 x,gy,dx,dw;x.shape={1,2};x.data={0.2f,-0.3f};gy.shape={1,2};gy.data={0.1f,-0.2f};
    kmeans::Config cfg;cfg.feature_dim=2;cfg.num_clusters=2;kmeans::Layer l;
    success(kmeans::new_configured(cfg,DType::Float32,weights,&l));success(kmeans::forward(l,x,&pre,&post));
    check(std::abs(post.data[0]+post.data[1]-1)<1e-6,"KMeans assignments not normalized");
    success(kmeans::backward_f32(l,gy,x,pre,&dx,&dw));success(kmeans::apply_grad_sgd(&l,dw,0.1));
    parallel::Config pc;pc.dim=2;pc.out_feat=2;pc.branches=2;pc.combine=parallel::Combine::Add;
    float packed[]={0.1f,0.3f,-0.2f,0.4f,-0.1f,0.2f,0.3f,0.1f};parallel::Layer pl;
    success(parallel::new_configured(pc,DType::Float32,packed,&pl));success(parallel::forward(pl,x,&pre,&post));
    success(parallel::backward_f32(pl,gy,x,pre,&dx,&dw));success(parallel::apply_grad_sgd(&pl,dw,0.1));
    check(dw.data.size()==8&&dx.data.size()==2,"Parallel gradient layout");
    float abi_pre[2],abi_post[2],abi_dx[2],abi_dw[8],abi_w[8];double abi_loss;
    auto status=welvet_parallel_normalbp_f32(1,1,2,2,2,1,0,1,0,0,packed,x.data.data(),gy.data.data(),0,0.1,abi_pre,abi_post,abi_dx,abi_dw,abi_w,&abi_loss);
    check(status.code==0,status.message);
    check(std::memcmp(abi_dx,dx.data.data(),sizeof(abi_dx))==0&&std::memcmp(abi_dw,dw.data.data(),sizeof(abi_dw))==0,"Parallel ABI gradients differ");
    float exported[8];success(parallel::export_packed_f32(pl,exported));
    check(std::memcmp(abi_w,exported,sizeof(exported))==0&&abi_loss==0,"Parallel ABI SGD differs");
  }
  // Exercise the newly enabled mesh SGD dispatcher for both added kinds.
  // These are GPU primitives; Go parity for shared 3D credit is separate.
  for (int kind=0;kind<2;++kind) {
    auto grid=architecture::new_grid(1,1,1,1);
    float initial[]={0.1f,0.3f,-0.2f,0.4f,-0.1f,0.2f,0.3f,0.1f};
    TensorF32 x,target;x.shape={1,2};x.data={0.2f,-0.3f};target.shape={1,2};target.data={0,1};
    if(kind==0){kmeans::Config cfg;cfg.feature_dim=2;cfg.num_clusters=2;kmeans::Layer layer;
      success(kmeans::new_configured(cfg,DType::Float32,initial,&layer));
      success(architecture::bind_kmeans(grid,0,0,0,0,std::move(layer)));
    }else{parallel::Config cfg;cfg.dim=2;cfg.out_feat=2;cfg.branches=2;cfg.combine=parallel::Combine::Add;parallel::Layer layer;
      success(parallel::new_configured(cfg,DType::Float32,initial,&layer));
      success(architecture::bind_parallel(grid,0,0,0,0,std::move(layer)));
    }
    double loss;success(runtime::step::train_mse(grid,x,target,1,0.1,&loss));
    check(std::isfinite(loss),"KMeans/Parallel mesh loss is not finite");
    float after[8];auto* cell=architecture::at(grid,0,0,0,0);
    if(kind==0)success(kmeans::export_packed_f32(cell->kmeans_op,after));
    else success(parallel::export_packed_f32(cell->parallel_op,after));
    check(std::memcmp(after,initial,(kind==0?4:8)*sizeof(float))!=0,"KMeans/Parallel mesh SGD did not update weights");
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

  // Subnormal updates must survive storage before a later phase reaches normal.
  float tiny_w[]={0,-0.0f,2e-38f,-2e-38f};
  float tiny_ref[]={0,-0.0f,2e-38f,-2e-38f};
  float tiny_g[]={1e-36f,-1e-36f,1e-36f,-1e-36f};
  for (int phase=0;phase<2;++phase) {
    for (int i=0;i<4;++i) {
      volatile float step=float(0.01*double(tiny_g[i]));
      tiny_ref[i]=tiny_ref[i]-step;
    }
    success(metal::sgd_f32(tiny_w,tiny_g,4,0.01));
    check(std::memcmp(tiny_w,tiny_ref,sizeof(tiny_w))==0,"Float32 tiny SGD lost stored bits");
    for (float& g:tiny_g) g*=3;
  }
  float tiny_wide_w[]={0}; double tiny_wide_g[]={1e-36};
  volatile float wide_step=float(0.01*tiny_wide_g[0]);
  float wide_ref=0.0f-wide_step;
  success(metal::sgd_f32_f64_grad(tiny_wide_w,tiny_wide_g,1,0.01));
  check(std::memcmp(tiny_wide_w,&wide_ref,sizeof(float))==0,"Wide-gradient tiny SGD lost stored bits");

  auto dense_pair=[]() {
    stack::Layer ops;ops.children.resize(2);
    for(int i=0;i<2;++i){ops.children[i].kind=stack::Kind::Dense;float initial[]={float(i+1)};
      success(dense::new_layer(1,1,Activation::Linear,DType::Float32,initial,&ops.children[i].dense_op));}
    return ops;
  };
  stack::Layer split_three;split_three.children.resize(3);
  for(auto& ch:split_three.children){ch.kind=stack::Kind::Dense;float one[]={1};
    success(dense::new_layer(1,1,Activation::Linear,DType::Float32,one,&ch.dense_op));}
  double mode_loss=0;
  success(runtime::tween_split_train::train_stack_mse(split_three,cx,ct,0.1,&mode_loss));
  volatile float split_step=float(0.1*double(float(2.0/3.0)));
  for(auto& ch:split_three.children) check(ch.dense_op.weights.f32[0]==1.0f-split_step,"Split did not share 1/N");
  auto alt_pair=dense_pair();
  success(runtime::tween_alt_train::train_stack_mse(alt_pair,cx,ct,0.1,&mode_loss));
  check(std::abs(mode_loss-2.0736)<1e-6 &&
        std::abs(alt_pair.children[0].dense_op.weights.f32[0]-0.656f)<1e-6 &&
        std::abs(alt_pair.children[1].dense_op.weights.f32[0]-1.6848f)<1e-6,
        "Alt did not refresh the gap between Split and Tween phases");
  // Distinct proxy update order: Linear uses updated W, Async saves post-update Wᵀgy.
  for(int variant=1;variant<=6;++variant){
    auto ops=dense_pair();double loss=0;
    success(systems::tween_split::train_stack_variant(ops,cx,ct,0.1,&loss,
              static_cast<systems::tween_split::Variant>(variant)));
    const float want0[]={0,0.2f,0.64f,0.2f,0.64f,0.36f,0.2f};
    const float want1[]={0,1.6f,1.8f,1.6f,1.8f,1.6f,1.6f};
    check(loss==4 && std::abs(ops.children[0].dense_op.weights.f32[0]-want0[variant])<1e-6 &&
          std::abs(ops.children[1].dense_op.weights.f32[0]-want1[variant])<1e-6,"Proxy credit/order mismatch");
  }
  // Strong semantic controls remain exact despite the CPU/GPU tolerance band.
  {parallel::Config cfg;cfg.dim=1;cfg.out_feat=1;cfg.branches=2;cfg.combine=parallel::Combine::Add;
    parallel::Layer adversarial;float w[]={1,1};success(parallel::new_configured(cfg,DType::Float32,w,&adversarial));
    adversarial.controls.modes={TrainMode::Adversarial,TrainMode::Freeze};double loss;
    success(runtime::cameral_train::train_mse(adversarial,cx,ct,TrainMode::NormalBP,.1,&loss));
    check(loss==4 && std::abs(adversarial.branches[0].dense_op.weights.f32[0]-1.4f)<1e-6 && adversarial.branches[1].dense_op.weights.f32[0]==1,"Adversarial branch failed ascent/frozen peer");
    parallel::Layer memory;success(parallel::new_configured(cfg,DType::Float32,w,&memory));memory.controls.modes={TrainMode::Memory,TrainMode::Freeze};memory.controls.surprise_thresh=5;
    success(runtime::cameral_train::train_mse(memory,cx,ct,TrainMode::NormalBP,.1,&loss));check(memory.branches[0].dense_op.weights.f32[0]==1,"Memory sleeping branch updated");
    memory.controls.surprise_thresh=3;success(runtime::cameral_train::train_mse(memory,cx,ct,TrainMode::NormalBP,.1,&loss));check(std::abs(memory.branches[0].dense_op.weights.f32[0]-.6f)<1e-6,"Memory awake branch failed to train");
    parallel::Layer shadow;float sw[]={1,3};success(parallel::new_configured(cfg,DType::Float32,sw,&shadow));shadow.controls.modes={TrainMode::Shadow,TrainMode::NormalBP};TensorF32 target=cx;target.data[0]=4;
    success(runtime::cameral_train::train_mse(shadow,cx,target,TrainMode::NormalBP,.1,&loss));check(loss==0 && shadow.branches[0].dense_op.weights.f32[0]==1 && std::abs(shadow.branches[1].dense_op.weights.f32[0]-2.6f)<1e-6,"Shadow teacher/KD mismatch");
    parallel::Layer scaled;success(parallel::new_configured(cfg,DType::Float32,w,&scaled));scaled.controls.modes={TrainMode::NormalBP,TrainMode::NormalBP};scaled.controls.branch_lrs={.5,0};success(runtime::cameral_train::train_mse(scaled,cx,ct,TrainMode::NormalBP,.1,&loss));check(std::abs(scaled.branches[0].dense_op.weights.f32[0]-.8f)<1e-6 && scaled.branches[1].dense_op.weights.f32[0]==1,"Branch LR scaling/zero mismatch");
  }
  for(auto mode:{TrainMode::Freeze,TrainMode::Shadow}){stack::Child frozen;frozen.kind=stack::Kind::Dense;float w[]={2};success(dense::new_layer(1,1,Activation::Linear,DType::Float32,w,&frozen.dense_op));TensorF32 pre,post,gy=cx,gx;gy.data[0]=4;success(stack::child_forward(frozen,cx,&pre,&post));success(runtime::cameral_train::train_child(frozen,gy,cx,pre,post,mode,.1,&gx));check(gx.data==std::vector<float>{8} && frozen.dense_op.weights.f32[0]==2,"Frozen input gradient did not pass through");}
  // CamSync must blend on GPU, respect one-way writes and invalidate caches.
  {dense::Layer teacher,student;float wa[]={1},wb[]={3};
    success(dense::new_layer(1,1,Activation::Linear,DType::Float32,wa,&teacher));
    success(dense::new_layer(1,1,Activation::Linear,DType::Float32,wb,&student));
    teacher.weights.bias={1};student.weights.bias={3};TensorF32 pre,post;
    success(dense::forward_f32(student,cx,&pre,&post));check(post.data[0]==6,"CamSync cache prime");
    success(cameral::blend_stores({{&teacher.weights,&teacher},{&student.weights,&student}},{0,1}));
    success(dense::forward_f32(student,cx,&pre,&post));
    check(teacher.weights.f32[0]==1 && teacher.weights.bias[0]==1 && student.weights.f32[0]==2 && student.weights.bias[0]==2 && post.data[0]==4,"CamSync one-way/bias/cache mismatch");
    auto rejected=cameral::blend_stores({{&teacher.weights,&teacher},{&student.weights,&student}},{1});
    check(!rejected && student.weights.f32[0]==2,"CamSync invalid alpha count mutated stores");
  }
  for(int dtype=0;dtype<kNumDTypes;++dtype){float wa[]={.2f,.4f},wb[]={-.3f,.1f};WeightStore a,b;
    success(WeightStore::make(1,2,DType(dtype),wa,&a));success(WeightStore::make(1,2,DType(dtype),wb,&b));
    auto before_a=a.native,before_b=b.native;auto fa=a.f32,fb=b.f32;float sa=a.scale,sb=b.scale;
    success(cameral::blend_stores({{&a},{&b}},{0,-1}));
    check(a.native==before_a && b.native==before_b && a.f32==fa && b.f32==fb && a.scale==sa && b.scale==sb,"CamSync no-write changed native payload");
  }
  for(int credit=0;credit<=9;++credit){
    auto ops=dense_pair();runtime::line::State pipe;
    success(runtime::line::train_mse(ops,&pipe,cx,ct,0.1,&mode_loss,runtime::line::Credit(credit)));
    check(mode_loss==0 && runtime::line::in_flight(pipe)==1,"Line failed warmup");
    success(runtime::line::train_mse(ops,&pipe,cx,ct,0.1,&mode_loss,runtime::line::Credit(credit)));
    const float want0[]={0.2f,0.8f,0.6f,0.4f,0.6f,0.6f,0.6f,0.6f,0.6f,0.6f},
                want1[]={1.6f,1.8f,1.6f,1.4f,1.6f,1.6f,1.6f,1.6f,1.6f,1.6f};
    check(mode_loss==4 && std::abs(ops.children[0].dense_op.weights.f32[0]-want0[credit])<1e-6 &&
      std::abs(ops.children[1].dense_op.weights.f32[0]-want1[credit])<1e-6,"Line used wrong credit or refreshed a frozen gap");
  }
  {auto ops=dense_pair();runtime::line::State pipe;runtime::line::TickResult tr;
    auto rejected=runtime::line::tick(ops,&pipe,cx,&ct,runtime::line::FlightRole::Train,0.1,&tr,
                                      static_cast<runtime::line::Credit>(10));
    check(!rejected && pipe.flights.empty(),"Unsupported Line mode mutated the pipeline");
    for(int i=0;i<2;++i)success(runtime::line::tick(ops,&pipe,cx,nullptr,runtime::line::FlightRole::Action,0.1,&tr,
                                                runtime::line::Credit::FamilyTween));
    check(!tr.popped_train && tr.action_out.data==std::vector<float>({2}) &&
          ops.children[0].dense_op.weights.f32[0]==1 && ops.children[1].dense_op.weights.f32[0]==2,
          "Action flight trained weights");
  }
  for(int credit=0;credit<=1;++credit){
    auto grid=architecture::new_grid(1,1,2,1);
    for(int i=0;i<2;++i){dense::Layer leaf;float initial[]={float(i+1)};
      success(dense::new_layer(1,1,Activation::Linear,DType::Float32,initial,&leaf));
      success(architecture::bind_dense(grid,0,0,i,0,std::move(leaf)));}
    success(runtime::step::train_mse(grid,cx,ct,2,0.1,&mode_loss,runtime::step::Credit(credit)));
    check(mode_loss==4 && std::abs(grid.cells[0].dense_op.weights.f32[0]-(credit?0.8f:0.2f))<1e-6 &&
      std::abs(grid.cells[1].dense_op.weights.f32[0]-(credit?1.8f:1.6f))<1e-6,"Mesh clock used wrong credit");
  }
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
