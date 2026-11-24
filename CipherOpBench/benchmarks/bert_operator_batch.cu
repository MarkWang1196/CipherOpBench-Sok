#include "cipheropbench/column_methods.cuh"
#include "cipheropbench/gpu_reservation.cuh"
#include "cipheropbench/kernels/matrix_mul.cuh"
#include "cipheropbench/literature.cuh"
#include "cipheropbench/literature_coefficients.cuh"
#include "cipheropbench/nexus_batch_protocol.cuh"
#include "cipheropbench/published.cuh"
#include "cipheropbench/research.cuh"
#include "../src/literature_math.cuh"
#include "../src/column_matrix.cuh"
#include "../src/research_coefficients.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <numbers>
#include <optional>
#include <set>

using namespace cipheropbench;
using Tensor = torch::Tensor;

namespace {
constexpr int tokens = 128, slots = 32768, capacity = 256;
constexpr double block_softmax_validation_atol = 1e-5;

Tensor polynomial(const Tensor& x, const std::vector<double>& coefficients) {
  auto y = torch::full_like(x, coefficients.back());
  for (int i = int(coefficients.size()) - 2; i >= 0; --i) y = y*x + coefficients[i];
  return y;
}

Tensor series(const Tensor& x, const std::vector<double>& coefficients) {
  auto b1 = torch::zeros_like(x), b2 = torch::zeros_like(x);
  for (int i = int(coefficients.size()) - 1; i > 0; --i) {
    auto b = 2*x*b1 - b2 + coefficients[i];
    b2 = b1;
    b1 = b;
  }
  return x*b1 - b2 + coefficients[0];
}

Tensor euston_reference(const Tensor& x,double bound) {
  const std::vector<double> g{0,5850./1024,0,-34974./1024,0,97015./1024,0,-113492./1024,0,46623./1024};
  const std::vector<double> f{0,315./128,0,-420./128,0,378./128,0,-180./128,0,35./128};
  auto s = x/bound;
  for (int i=0; i<3; ++i) s = polynomial(s,g);
  for (int i=0; i<3; ++i) s = polynomial(s,f);
  auto exponential = [](Tensor z) {
    auto y = polynomial(z/8.,cipheropbench::literature::EUSTON_EXP);
    return y.pow(8);
  };
  auto a = x*s;
  auto d = 1. + exponential(-1.702*a);
  return a*s*exponential(.851*(s-1.)*a)*(1.-(1.-d).pow(16))/d;
}

Tensor nexus_gelu_reference(const Tensor& x) {
  const std::vector<double> g{0,5850./1024,0,-34974./1024,0,97015./1024,0,-113492./1024,0,46623./1024};
  const std::vector<double> f{0,315./128,0,-420./128,0,378./128,0,-180./128,0,35./128};
  auto sign=[&](double shift){
    auto value=(x+shift)/nexus_batch::gelu_sign_divisor;
    value=.5*polynomial(polynomial(value,g),g);
    return .5*polynomial(polynomial(value,f),f);
  };
  auto low=sign(4.),mid=sign(1.95),high=sign(-3.);
  auto clip=x*(low-high)+4.*low+3.*high-.5;
  return polynomial(clip,nexus_batch::gelu_lower)*(low-mid)+
      polynomial(clip,nexus_batch::gelu_upper)*(mid-high)+x*(high+.5);
}

struct EustonFfnProfile { double ln1=0.,gelu=16.,ln2=0.; };

EustonFfnProfile euston_ffn_profile(const char* file) {
  std::ifstream in(file);
  std::string magic,task,model,revision;
  EustonFfnProfile profile;
  in>>magic>>task>>model>>revision>>profile.ln1>>profile.gelu>>profile.ln2;
  TORCH_CHECK(in&&magic=="CIPHEROPBENCH_EUSTON_BATCH_FFN_V1"&&task=="sst2"&&
              std::isfinite(profile.ln1)&&profile.ln1>0.&&std::isfinite(profile.ln2)&&profile.ln2>0.&&
              std::isfinite(profile.gelu)&&profile.gelu>=1.&&profile.gelu<=64.,
              "Invalid Euston public FFN profile");
  TORCH_CHECK(!(in>>magic),"Unexpected trailing Euston FFN profile fields");
  return profile;
}

Tensor diagonals(const Tensor& x, bool giant_shifted=false) {
  auto y = torch::empty_like(x);
  for (int t=0; t<tokens; ++t) y.select(1,t).copy_(torch::roll(x.select(1,t),-t,1));
  if (giant_shifted) {
    const int baby=cipheropbench::detail::column_baby;
    for (int d=baby; d<tokens; ++d) {
      auto column=y.select(2,d);
      column.copy_(torch::roll(column,d/baby*baby,1));
    }
  }
  return y;
}

Tensor softmax_reference(Tensor scores, const Tensor& valid, const calibration::Layer& profile) {
  auto mask = valid.unsqueeze(1) & valid.unsqueeze(2);
  const double power=std::exp2(profile.exp_squarings);
  auto p=(1. + (scores-profile.softmax_shift)/power).pow(power).masked_fill(~mask,0.);
  auto sum=p.sum(-1,true).masked_fill(~valid.unsqueeze(-1),1.)+1e-5;
  TORCH_CHECK(sum.min().item<double>()>0.&&sum.max().item<double>()<2.,"MOAI row denominator outside the Goldschmidt interval");
  return diagonals(p*(1.-(1.-sum).pow(131072))/sum);
}

std::array<EustonSoftmaxProfile,12> batch_softmax_profiles(const char* file,bool nexus) {
  std::ifstream in(file);
  std::string magic,task,model,revision;
  int heads=0;
  in>>magic>>task>>model>>revision>>heads;
  TORCH_CHECK(in&&magic==(nexus?"CIPHEROPBENCH_NEXUS_BATCH_SOFTMAX_V1":"CIPHEROPBENCH_EUSTON_BATCH_SOFTMAX_V1")&&
              task=="sst2"&&heads==12,"Invalid public Softmax profile");
  std::array<EustonSoftmaxProfile,12> profiles;
  for(auto& p:profiles) {
    in>>p.shift>>p.squarings>>p.divisor>>p.inverse_iterations;
    TORCH_CHECK(in&&std::isfinite(p.shift)&&std::isfinite(p.divisor)&&p.divisor>0&&
                (nexus?p.squarings==8:p.squarings>=0&&p.squarings<=3)&&
                p.inverse_iterations>=0&&p.inverse_iterations<=24,"Invalid Softmax head parameters");
  }
  TORCH_CHECK(!(in>>magic),"Unexpected trailing Euston profile fields");
  return profiles;
}

Tensor euston_softmax_reference(const Tensor& scores,const Tensor& valid,const EustonSoftmaxProfile& profile) {
  auto pair=valid.unsqueeze(1)&valid.unsqueeze(2);
  const double power=std::exp2(profile.squarings);
  auto z=(scores-profile.shift)/power;
  auto active=z.masked_select(pair);
  TORCH_CHECK(active.min().item<double>()>-5.&&active.max().item<double>()<0.,
              "Euston exponential inputs outside (-5,0)");
  auto p=(polynomial(z,cipheropbench::literature::EUSTON_EXP).pow(power)/profile.divisor).masked_fill(~pair,0.);
  auto sum=p.sum(-1,true).masked_fill(~valid.unsqueeze(-1),1.);
  TORCH_CHECK(sum.min().item<double>()>0.&&sum.max().item<double>()<2.,
              "Euston normalized denominator outside (0,2)");
  const double repeats=std::exp2(profile.inverse_iterations+1);
  return diagonals(p*(1.-(1.-sum).pow(repeats))/sum);
}

struct Timing {
  int warmup;
  std::vector<double> seconds;
  double max_error=0., max_mae=0.;
  bool validated=true;
  int logical_samples=0;
  Tensor exact_reference, error_mask;
  double exact_error_sum=0.;
  int64_t exact_error_count=0;

  template<class Evaluate,class Decode>
  void measure(Runtime& r, Evaluate evaluate, Decode decode, const Tensor& expected,
               const std::function<void()>& prepare={}, double atol=1e-2, double rtol=1e-3) {
    TORCH_CHECK(torch::isfinite(expected).all().item<bool>(), "Nonfinite method reference");
    for (int i=-warmup; i<int(seconds.size()); ++i) {
      if(prepare) prepare();
      r.synchronize();
      auto start=std::chrono::steady_clock::now();
      auto output=evaluate();
      r.synchronize();
      auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
      Tensor result=decode(output);
      TORCH_CHECK(result.sizes()==expected.sizes(),"Decoded batch geometry differs from reference");
      Tensor error=(result-expected).abs();
      bool finite=torch::isfinite(error).all().item<bool>();
      if (finite) {
        max_error=std::max(max_error,error.max().item<double>());
        max_mae=std::max(max_mae,error.mean().item<double>());
      }
      validated=finite&&torch::allclose(result,expected,rtol,atol)&&max_mae<1e-3;
      if (!validated) {
        if(finite){
          auto excess=error-(atol+expected.abs()*rtol);
          const auto worst=excess.flatten().argmax().item<int64_t>();
          std::cerr<<"Validation outside timer: max_excess="<<excess.flatten()[worst].item<double>()
                   <<" expected="<<expected.flatten()[worst].item<double>()
                   <<" actual="<<result.flatten()[worst].item<double>()<<" mae="<<max_mae<<'\n';
        }
        return;
      }
      if (i>=0) {
        seconds[i]+=elapsed;
        if(logical_samples) {
          auto target=exact_reference.defined()?exact_reference:expected;
          auto errors=(result.narrow(0,0,logical_samples)-target.narrow(0,0,logical_samples)).abs();
          if(error_mask.defined())
            errors=errors.masked_select(error_mask.narrow(0,0,logical_samples).expand_as(errors));
          TORCH_CHECK(errors.numel()>0&&torch::isfinite(errors).all().item<bool>(),"Invalid single-request error");
          exact_error_sum+=errors.sum().item<double>();
          exact_error_count+=errors.numel();
        }
      }
    }
  }
};

Tensor project_heads(const Tensor& input,const LinearWeights& weights) {
  return (torch::matmul(input,weights.weight)+weights.bias)
      .reshape({input.size(0),input.size(1),12,64}).transpose(1,2);
}

Tensor attention_values(const Tensor& hidden,const Tensor& valid,const LayerWeights& w) {
  auto q=project_heads(hidden,w.q), key=project_heads(hidden,w.key), value=project_heads(hidden,w.v);
  auto scores=torch::matmul(q,key.transpose(-1,-2))/8.;
  auto probabilities=torch::softmax(scores.masked_fill(~valid.unsqueeze(1).unsqueeze(1),-100.),-1);
  return torch::matmul(probabilities,value).transpose(1,2).reshape(hidden.sizes());
}

CipherList encrypt_nexus_pair(Runtime& runtime,const Tensor& input,int levels) {
  TORCH_CHECK(input.dim()==3&&input.size(0)==2&&input.size(1)==tokens&&input.size(2)%tokens==0,
              "Expected two full NEXUS row matrices");
  CipherList result;
  for(int col=0;col<input.size(2);col+=tokens)
    result.push_back(runtime.encrypt(k::flatten_pack(input[0].narrow(1,col,tokens),input[1].narrow(1,col,tokens)),
                                     k::chain_idx(levels)));
  return result;
}

Tensor decrypt_nexus_pair(Runtime& runtime,const CipherList& output) {
  std::vector<Tensor> blocks;
  for(const auto& block:output)blocks.push_back(runtime.decrypt_slots(block));
  return torch::cat(blocks,2);
}

void measure_nexus_projection(int levels,Runtime& runtime,const Tensor& input,
                              const std::vector<const LinearWeights*>& projections,Timing& timing,int& features) {
  kernels::MMEvaluator kernel(runtime.ckks);
  std::vector<CipherList> encrypted;
  for(int b=0;b<input.size(0);b+=2)encrypted.push_back(encrypt_nexus_pair(runtime,input.narrow(0,b,2),levels));
  for(const auto* weights:projections){
    k::FlatVecMat blocks;
    k::FlatVecArray bias;
    auto expected=torch::matmul(input,weights->weight)+weights->bias;
    for(int col=0;col<weights->weight.size(1);col+=tokens){
      auto& column=blocks.emplace_back();
      for(int row=0;row<weights->weight.size(0);row+=tokens){
        auto tile=weights->weight.narrow(0,row,tokens).narrow(1,col,tokens);
        column.push_back(k::flatten_pack(tile,tile));
      }
      bias.push_back(k::row_pack_128x1(weights->bias.narrow(0,col,tokens)));
    }
    for(size_t pair=0;pair<encrypted.size()&&timing.validated;++pair){
      timing.measure(runtime,[&]{return kernel.matrix_mul_paired_blocks(encrypted[pair],blocks,bias);},
                     [&](const auto& out){return decrypt_nexus_pair(runtime,out);},expected.narrow(0,2*pair,2));
      if(!timing.validated)
        std::cerr<<"NEXUS projection input max_error="
                 <<(decrypt_nexus_pair(runtime,encrypted[pair])-input.narrow(0,2*pair,2)).abs().max().item<double>()<<'\n';
      std::cerr<<"NEXUS projection sample pair "<<pair+1<<'/'<<encrypted.size()
               <<" validated="<<timing.validated<<" max_error="<<timing.max_error<<'\n';
    }
    if(!timing.validated)return;
    features+=expected.size(-1);
  }
}

void measure_column_projection(const std::string& stage,Runtime& runtime,ColumnOperators& columns,
                               const Tensor& hidden,const Tensor& valid,const LayerWeights& w,
                               Timing& timing,int& checked_features,PublishedOperators* block_ops,int nexus_levels) {
  auto keep=valid.unsqueeze(-1);
  Tensor input=hidden;
  std::vector<const LinearWeights*> projections{&w.q,&w.key,&w.v};
  if(stage!="qkv") {
    auto attention=attention_values(hidden,valid,w);
    if(stage=="wo") {
      input=std::move(attention);
      projections={&w.out};
    } else {
      auto residual=torch::matmul(attention,w.out.weight)+w.out.bias+hidden;
      input=torch::layer_norm(residual,{768},w.ln1_weight,w.ln1_bias,1e-12);
      projections={&w.up};
      if(stage=="down") {
        input=torch::gelu(torch::matmul(input,w.up.weight)+w.up.bias);
        projections={&w.down};
      }
    }
  }
  input=input.masked_fill(~keep,0.);
  if(nexus_levels){
    measure_nexus_projection(nexus_levels,runtime,input,projections,timing,checked_features);
    return;
  }
  auto encrypted=block_ops?block_ops->encrypt_block_batch(input):columns.encrypt_batch(input,k::chain_idx(8));
  cipheropbench::detail::Math math{runtime};math.prime_scale=true;math.column_token_stride=capacity;
  for(const auto* weights:projections) {
    auto expected=torch::matmul(input,weights->weight)+weights->bias;
    timing.measure(runtime,[&]{return block_ops?block_ops->batch_block_linear(encrypted,*weights):
                                            cipheropbench::detail::column_cpmm(math,encrypted,*weights);},
                   [&](const CipherList& output){return block_ops?block_ops->decrypt_block_batch(output):
                                                             columns.decrypt_batch(output,input.size(0));},expected);
    if(!timing.validated) break;
    checked_features+=expected.size(-1);
    std::cerr<<"Projection features checked: "<<checked_features<<" max_error="<<timing.max_error<<'\n';
  }
}

std::pair<Tensor,Tensor> column_residual_inputs(int kind,const Tensor& hidden,const Tensor& valid,const LayerWeights& w) {
  auto attention=torch::matmul(attention_values(hidden,valid,w),w.out.weight)+w.out.bias;
  if(kind==1) return {hidden,attention};
  auto first=torch::layer_norm(attention+hidden,{768},w.ln1_weight,w.ln1_bias,1e-12);
  auto activation=torch::gelu(torch::matmul(first,w.up.weight)+w.up.bias);
  return {first,torch::matmul(activation,w.down.weight)+w.down.bias};
}

Tensor ffn_trace_preactivation(const Tensor& hidden,const Tensor& valid,const LayerWeights& w,
                               int offset,int width) {
  auto pre=torch::zeros({hidden.size(0),hidden.size(1),width},hidden.options());
  const auto weight=w.up.weight.slice(1,offset,offset+width);
  const auto bias=w.up.bias.slice(0,offset,offset+width);
  for(int sample=0;sample<hidden.size(0);++sample){
    // Padding slots stay zero. Prepare only real tokens before the GPU timer.
    auto indices=valid[sample].nonzero().flatten();
    if(indices.numel()==0)continue;
    auto state=hidden.narrow(0,sample,1).index_select(1,indices);
    auto keep=valid.narrow(0,sample,1).index_select(1,indices);
    auto [left,right]=column_residual_inputs(1,state,keep,w);
    auto normalized=torch::layer_norm(left+right,{768},w.ln1_weight,w.ln1_bias,1e-12);
    pre[sample].index_copy_(0,indices,(torch::matmul(normalized,weight)+bias)[0]);
  }
  return pre;
}

std::vector<double> interleaved_mask(const Tensor& valid) {
  std::vector<double> mask(slots);
  auto keep=valid.accessor<bool,2>();
  for(int t=0;t<tokens;++t)for(int b=0;b<valid.size(0);++b)mask[t*capacity+b]=keep[b][t];
  return mask;
}

Tensor newton_goldschmidt_reference(const Tensor& variance) {
  auto estimate=-1.29054537e-4*variance+.129054537;
  for(int j=0;j<4;++j)estimate=estimate*(1.5-.5*variance*estimate.square());
  auto root=variance*estimate,half=.5*estimate;
  for(int j=0;j<2;++j){
    auto correction=.5-root*half;
    root=root+root*correction;
    half=half+half*correction;
  }
  return 2.*half;
}

void measure_column_norm(int kind,int input_levels,Runtime& runtime,ColumnOperators& columns,
                         const Tensor& hidden,const Tensor& valid,const LayerWeights& w,Timing& timing,
                         LiteratureOperators* euston,double divisor,const std::string& method) {
  auto [left,right]=column_residual_inputs(kind,hidden,valid,w);
  auto input=left+right;
  const auto& weight=kind==1?w.ln1_weight:w.ln2_weight;
  const auto& bias=kind==1?w.ln1_bias:w.ln2_bias;
  auto keep=valid.unsqueeze(-1);
  Tensor expected;
  std::vector<double> slot_mask;
  if(method=="blockformer"){
    PublishedOperators ops(runtime,method);
    input=input.masked_fill(~keep,0.);
    auto encrypted=ops.encrypt_block_batch(input,input_levels);
    auto expected=input-input.mean(-1,true)+bias;
    if(timing.logical_samples) {
      timing.exact_reference=torch::layer_norm(input,{768},weight,bias,1e-12);
      timing.error_mask=keep;
    }
    CipherList working;
    timing.measure(runtime,[&]{return ops.batch_block_norm(std::move(working),bias);},
        [&](const CipherList& out){return ops.decrypt_block_batch(out);},expected,[&]{working=encrypted;});
    return;
  }
  if(method=="nexus"){
    input=input.masked_fill(~keep,0.);
    auto centered=input-input.mean(-1,true);
    auto variance=centered.square().mean(-1,true);
    TORCH_CHECK(variance.max().item<double>()<=divisor,"NEXUS variance exceeds the public refresh bound");
    expected=centered*newton_goldschmidt_reference(variance)*weight+bias;
    k::FlatVecArray gamma,beta;
    for(int i=0;i<768;i+=tokens){
      gamma.push_back(k::row_pack_128x1(weight.narrow(0,i,tokens)));
      beta.push_back(k::row_pack_128x1(bias.narrow(0,i,tokens)));
    }
    kernels::LNEvaluator norm(runtime.ckks,runtime.bootstrapper);
    for(int sample=0;sample<hidden.size(0)&&timing.validated;sample+=2){
      auto encrypted=encrypt_nexus_pair(runtime,input.narrow(0,sample,2),input_levels);
      timing.measure(runtime,[&]{
        CipherList result;
        norm.layer_norm_paired_128x768(encrypted,result,gamma,beta,divisor);
        return result;
      },[&](const CipherList& out){return decrypt_nexus_pair(runtime,out);},expected.narrow(0,sample,2));
    }
    return;
  }
  if(euston) {
    auto centered=input-input.mean(-1,true);
    auto scaled=centered.square().sum(-1,true)/divisor;
    auto active=scaled.masked_select(keep);
    TORCH_CHECK(active.min().item<double>()>=0.&&active.max().item<double>()<2.,
                "Euston LayerNorm scaled variance outside [0,2)");
    auto t=scaled-1.;
    auto root=1.+.5*t-.125*t.square()+.0625*t.pow(3);
    auto inverse=(1.-(1.-root*.5).pow(128))/root;
    expected=(centered*inverse*std::sqrt(768./divisor)*weight+bias).masked_fill(~keep,0.);
    slot_mask=interleaved_mask(valid);
  } else if(method=="moai") {
    auto centered=input-input.mean(-1,true);
    auto variance=centered.square().sum(-1,true)/(kind==1?1.:768.);
    auto active=variance.masked_select(keep);
    TORCH_CHECK(active.min().item<double>()>=0.&&active.max().item<double>()<=(kind==1?1000.:16.),
                "MOAI LayerNorm variance exceeds the public batch bound");
    expected=(centered*newton_goldschmidt_reference(variance)*(kind==1?std::sqrt(768.):1.)*weight+bias).masked_fill(~keep,0.);
  } else {
    expected=torch::layer_norm(input,{768},weight,bias,1e-12).masked_fill(~keep,0.);
  }
  auto encrypted=columns.encrypt_batch(input.masked_fill(~keep,0.),k::chain_idx(input_levels));
  CipherList working;
  timing.measure(runtime,[&]{
                   if(euston)return euston->batch_layer_norm(std::move(working),weight,bias,slot_mask,divisor);
                   return columns.batch_layer_norm(std::move(working),weight,bias,valid,kind);
                 },
                 [&](const CipherList& output){return columns.decrypt_batch(output,input.size(0));},
                 expected,[&]{working=encrypted;});
}

void measure_column_residual(int kind,Runtime& runtime,ColumnOperators& columns,
                             const Tensor& hidden,const Tensor& valid,const LayerWeights& w,Timing& timing,
                             const std::string& method) {
  auto [left,right]=column_residual_inputs(kind,hidden,valid,w);
  auto keep=valid.unsqueeze(-1);
  left=left.masked_fill(~keep,0.);
  right=right.masked_fill(~keep,0.);
  cipheropbench::detail::Math m{runtime};
  auto add=[&](const CipherList& a,const CipherList& b){
    CipherList out;
    out.reserve(a.size());
    for(size_t i=0;i<a.size();++i)out.push_back(m.add(a[i],b[i]));
    return out;
  };
  if(method=="nexus"){
    for(int offset=0;offset<hidden.size(0)&&timing.validated;offset+=2){
      auto l=left.narrow(0,offset,2),r=right.narrow(0,offset,2);
      auto a=encrypt_nexus_pair(runtime,l,1),b=encrypt_nexus_pair(runtime,r,1);
      timing.measure(runtime,[&]{return add(a,b);},[&](const CipherList& out){return decrypt_nexus_pair(runtime,out);},l+r);
    }
    return;
  }
  if(method=="blockformer"){
    PublishedOperators ops(runtime,method);
    auto a=ops.encrypt_block_batch(left,6),b=ops.encrypt_block_batch(right,6);
    timing.measure(runtime,[&]{return add(a,b);},
        [&](const CipherList& out){return ops.decrypt_block_batch(out);},left+right);
    return;
  }
  auto a=columns.encrypt_batch(left,k::chain_idx(6));
  auto b=columns.encrypt_batch(right,k::chain_idx(6));
  timing.measure(runtime,[&]{return add(a,b);},
      [&](const CipherList& output){return columns.decrypt_batch(output,hidden.size(0));},left+right);
}

int nexus_projection(const LinearWeights& weights,int warmup,int runs,const char* output,Tensor input,int tile){
  constexpr size_t degree=8192,rows=4096,hidden=768,width=64;
  const bool online=input.defined();
  reserve_gpu_memory();
  phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
  parms.set_poly_modulus_degree(degree);
  parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(degree,{60,40,60}));
  auto context=std::make_shared<PhantomContext>(parms);
  auto secret=std::make_shared<PhantomSecretKey>(*context);
  auto pub=std::make_shared<PhantomPublicKey>(secret->gen_publickey(*context));
  auto relin=std::make_shared<PhantomRelinKey>(secret->gen_relinkey(*context));
  auto encoder=std::make_shared<PhantomCKKSEncoder>(*context);
  std::vector<uint32_t> elements;
  for(size_t i=1;i<degree;i*=2)elements.push_back(degree/i+1);
  auto rotations=std::make_shared<PhantomGaloisKey>(secret->create_galois_keys_from_elts(*context,elements));
  auto ckks=std::make_shared<nexus::CKKSEvaluator>(context,pub,secret,encoder,relin,rotations,k::scale,elements);
  std::optional<PhantomSecretKey> client;
  if(online)client.emplace(*context);
  kernels::MMEvaluator kernel(ckks);
  torch::manual_seed(20261109);
  const int first=64*(tile%12);
  auto matrix=weights.weight.slice(1,first,first+width).transpose(0,1).contiguous();
  auto bias=weights.bias.slice(0,first,first+width).unsqueeze(-1);
  if(online)input=input.reshape({rows,hidden}).transpose(0,1).contiguous();
  std::vector<std::vector<double>> packed;
  const auto* values=matrix.data_ptr<double>();
  for(size_t i=0;i<hidden*width;i+=degree)packed.emplace_back(values+i,values+i+degree);
  auto sync=[] {auto code=cudaDeviceSynchronize();TORCH_CHECK(code==cudaSuccess,cudaGetErrorString(code));};
  std::vector<double> seconds, offline_seconds, online_seconds;
  double max_error=0.;
  for(int run=-warmup;run<runs;++run){
    auto masks=(torch::rand({hidden,rows},torch::kDouble)-.5)*.2;
    auto expected=online?torch::matmul(matrix,input)+bias:torch::matmul(matrix,masks);
    auto truth=expected.contiguous();
    auto expected_values=truth.accessor<double,2>();
    auto inputs=masks.accessor<double,2>();
    std::vector<std::vector<double>> clear;
    for(size_t i=0;i<hidden;++i)clear.emplace_back(inputs[i].data(),inputs[i].data()+rows);
    CipherList result;
    sync();const auto start=std::chrono::steady_clock::now();
    kernel.matrix_mul(clear,packed,result);
    TORCH_CHECK(result.size()==width,"NEXUS offline output width differs");
    if(online){
      for(auto& col:result)nexus_batch::add_client_share(col,*client,*context,ckks->encoder);
      for(auto& col:result)nexus_batch::remove_server_share(col,*secret,*context);
    }
    sync();const double offline=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    double online_time=0.;
    if(online){
      auto masked=input-masks;
      const auto online_start=std::chrono::steady_clock::now();
      auto correction=(2.*(torch::matmul(matrix,masked)+bias)).contiguous();
      auto columns=correction.accessor<double,2>();
      for(size_t f=0;f<width;++f){
        std::vector<double> values(columns[f].data(),columns[f].data()+rows);
        auto pt=k::CKKSEncode(std::move(values),ckks,&result[f]);
        ckks->evaluator.add_plain_inplace(result[f],pt);
      }
      sync();
      online_time=std::chrono::duration<double>(std::chrono::steady_clock::now()-online_start).count();
    }
    for(size_t f=0;f<width;++f){
      PhantomPlaintext p;std::vector<double> slots;
      if(online)client->decrypt(*context,result[f],p);
      else ckks->decryptor.decrypt(result[f],p);
      ckks->encoder.decode(p,slots);
      TORCH_CHECK(slots.size()==rows,"NEXUS offline slot count differs");
      for(size_t i=0;i<rows;++i){
        const auto error=std::abs(slots[i]/2.-expected_values[f][i]);
        TORCH_CHECK(std::isfinite(error)&&error<=.01,"NEXUS batch projection mismatch at sample ",i/128,", feature ",f);
        max_error=std::max(max_error,error);
      }
    }
    if(run>=0){
      seconds.push_back(offline+online_time);
      offline_seconds.push_back(offline);
      if(online)online_seconds.push_back(online_time);
    }
  }
  std::ofstream file(output,std::ios::app);TORCH_CHECK(file.good(),"Cannot open NEXUS batch result");
  file<<std::setprecision(14)<<"{\"implementation\":\"cipheropbench_common_library\",\"method\":\"NEXUS\",\"scope\":\""
      <<(online?"nexus_input_projection_tile":"nexus_offline_pcmm")
      <<"\",\"active_samples\":32,\"sequence_length\":128,\"N\":8192,\"input\":\""
      <<(online?"distinct_model_traces":"synthetic_offline_mask_checkpoint_weights")<<"\","
      <<"\"warmup\":"<<warmup<<",\"runs\":"<<runs<<",\"parts_checked\":64,\"output_features\":64,\"max_abs_error\":"
      <<max_error<<",\"status\":\"validated\"";
  auto write_times=[&](const char* name,const auto& times){
    file<<",\""<<name<<"\":[";
    for(size_t i=0;i<times.size();++i)file<<(i?",":"")<<times[i];
    file<<']';
  };
  write_times("batch_seconds",seconds);
  if(online){
    file<<",\"online_clear_mm_device\":\"cpu\",\"key_transfer\":\"commutable_encryption_two_keys\"";
    file<<",\"feature_offset\":"<<width*tile;
    write_times("offline_batch_seconds",offline_seconds);
    write_times("online_batch_seconds",online_seconds);
  }
  file<<"}\n";return 0;
}

int euston_initial_qkv(const Tensor& input,const LayerWeights& w,const char* output){
  constexpr int degree=8192, group=32, rows=group*tokens, width=3*768;
  TORCH_CHECK(input.size(0)%group==0,"Euston IWMM requires full 32-sample ciphertext groups");
  reserve_gpu_memory();
  phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
  parms.set_poly_modulus_degree(degree);
  parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(degree,{60,40,40,40,60}));
  auto context=std::make_shared<PhantomContext>(parms);
  auto secret=std::make_shared<PhantomSecretKey>(*context);
  auto pub=std::make_shared<PhantomPublicKey>(secret->gen_publickey(*context));
  auto relin=std::make_shared<PhantomRelinKey>(secret->gen_relinkey(*context));
  auto encoder=std::make_shared<PhantomCKKSEncoder>(*context);
  std::vector<uint32_t> elements;
  for(int i=1;i<degree;i*=2)elements.push_back(degree/i+1);
  auto rotations=std::make_shared<PhantomGaloisKey>(secret->create_galois_keys_from_elts(*context,elements));
  auto ckks=std::make_shared<nexus::CKKSEvaluator>(context,pub,secret,encoder,relin,rotations,k::scale,elements);
  kernels::MMEvaluator kernel(ckks);
  auto& ev=ckks->evaluator;
  auto sync=[] {auto code=cudaDeviceSynchronize();TORCH_CHECK(code==cudaSuccess,cudaGetErrorString(code));};
  auto weights=torch::cat({w.q.weight,w.key.weight,w.v.weight},1);
  auto bias=torch::cat({w.q.bias,w.key.bias,w.v.bias});
  std::vector<double> offline,online;
  double max_error=0.,max_mae=0.;
  torch::manual_seed(20261110);
  for(int start=0;start<input.size(0);start+=group){
    auto hidden=input.narrow(0,start,group);
    auto expected=torch::matmul(hidden,weights)+bias;
    sync();
    const auto offline_start=std::chrono::steady_clock::now();
    auto mask=2.*torch::rand_like(hidden)-1.;
    auto [u,s,vh]=torch::linalg_svd(mask,false);
    auto singular=s.contiguous();
    std::vector<double> coefficients(degree);
    std::copy_n(singular.data_ptr<double>(),rows,coefficients.begin());
    Cipher compressed;
    kernel.enc_compress_ciphertext(coefficients,compressed);
    auto diagonal=kernel.decompress_ciphertext(compressed);
    std::vector<PhantomPlaintext> selectors;
    selectors.reserve(group);
    for(int b=0;b<group;++b){
      std::vector<double> select(rows);
      std::fill_n(select.begin()+b*tokens,tokens,1.);
      selectors.push_back(k::CKKSEncode(std::move(select),ckks,&compressed,true));
    }
    auto packed_u=u.permute({2,0,1}).contiguous().reshape({tokens,rows});
    auto column_values=packed_u.accessor<double,2>();
    CipherList e(tokens);
    for(int j=0;j<tokens;++j){
      for(int b=0;b<group;++b){
        auto& d=diagonal[b*tokens+j];
        d.set_scale(d.scale()*degree);
        Cipher term;
        ev.multiply_plain(d,selectors[b],term);
        if(b==0)e[j]=std::move(term);else ev.add_inplace(e[j],term);
      }
      ev.rescale_to_next_inplace(e[j]);
      auto pt=k::CKKSEncode(std::vector<double>(column_values[j].data(),column_values[j].data()+rows),ckks,&e[j],true);
      ev.multiply_plain_inplace(e[j],pt);
      ev.rescale_to_next_inplace(e[j]);
    }
    diagonal.clear();
    auto right=torch::matmul(vh,weights).contiguous();
    auto h=right.accessor<double,3>();
    CipherList result(width);
    for(int f=0;f<width;++f){
      for(int j=0;j<tokens;++j){
        std::vector<double> lane(rows);
        for(int b=0;b<group;++b)std::fill_n(lane.begin()+b*tokens,tokens,h[b][j][f]);
        auto pt=k::CKKSEncode(std::move(lane),ckks,&e[j],true);
        Cipher term;
        ev.multiply_plain(e[j],pt,term);
        if(j==0)result[f]=std::move(term);else ev.add_inplace(result[f],term);
      }
      ev.rescale_to_next_inplace(result[f]);
    }
    sync();
    offline.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-offline_start).count());
    const auto online_start=std::chrono::steady_clock::now();
    auto correction=(torch::matmul(hidden-mask,weights)+bias).reshape({rows,width}).t().contiguous();
    auto plain=correction.accessor<double,2>();
    for(int f=0;f<width;++f){
      auto pt=k::CKKSEncode(std::vector<double>(plain[f].data(),plain[f].data()+rows),ckks,&result[f]);
      ev.add_plain_inplace(result[f],pt);
    }
    sync();
    online.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-online_start).count());
    auto actual=torch::empty({rows,width},torch::kDouble);
    for(int f=0;f<width;++f){
      auto decoded=k::CKKSDecrypt(result[f],ckks);
      TORCH_CHECK(decoded.size()==rows,"Euston IWMM slot count differs");
      actual.select(1,f).copy_(torch::from_blob(decoded.data(),{rows},torch::kDouble));
    }
    expected=expected.reshape_as(actual);
    auto error=(actual-expected).abs();
    max_error=std::max(max_error,error.max().item<double>());
    max_mae=std::max(max_mae,error.mean().item<double>());
    TORCH_CHECK(torch::allclose(actual,expected,1e-3,1e-2)&&max_mae<1e-3,
                "Euston IWMM validation failed in group ",start/group," max_error=",max_error," max_mae=",max_mae);
    std::cerr<<"Euston IWMM samples checked: "<<start+group<<'/'<<input.size(0)<<" max_error="<<max_error<<'\n';
  }
  const double off=std::accumulate(offline.begin(),offline.end(),0.);
  const double on=std::accumulate(online.begin(),online.end(),0.);
  std::ofstream file(output,std::ios::app);TORCH_CHECK(file.good(),"Cannot open Euston batch result");
  file<<std::setprecision(14)<<"{\"implementation\":\"cipheropbench_common_library\",\"method\":\"EUSTON\","
      <<"\"scope\":\"euston_initial_qkv\",\"active_samples\":"<<input.size(0)
      <<",\"sequence_length\":128,\"N\":8192,\"input\":\"distinct_model_traces\","
      <<"\"profile\":\"published_svd_iwmm\",\"layout\":\"sample_major_full_columns\","
      <<"\"physical_group_size\":32,\"measured_group_count\":"<<offline.size()
      <<",\"warmup\":0,\"runs\":1,\"parts_checked\":2304,\"output_features\":2304,"
      <<"\"max_abs_error\":"<<max_error<<",\"max_mae\":"<<max_mae<<",\"status\":\"validated\","
      <<"\"offline_batch_seconds\":["<<off<<"],\"online_batch_seconds\":["<<on
      <<"],\"batch_seconds\":["<<off+on<<"]}\n";
  return 0;
}
}

int main(int argc, char** argv) {
  try {
    TORCH_CHECK(argc==9||argc==10, "Usage: cipheropbench_operator_batch SCOPE BATCH WARMUP RUNS OUTPUT FIXTURES PROFILE_OR_DASH WIDTH_OR_HEADS [OFFSET]");
    std::string scope=argv[1];
    int active=std::stoi(argv[2]), warmup=std::stoi(argv[3]), runs=std::stoi(argv[4]), limit=std::stoi(argv[8]);
    const bool euston_av=scope=="euston_av", euston_qk=scope=="euston_qk";
    const bool moai_softmax=scope=="moai_softmax", euston_softmax=scope=="euston_softmax";
    const bool arion_softmax=scope=="arion_softmax",nexus_softmax=scope=="nexus_softmax";
    const bool blockformer_softmax=scope=="blockformer_softmax";
    const bool euston_gelu_trace=scope=="euston_gelu_trace", arion_gelu=scope=="arion_gelu", moai_gelu=scope=="moai_gelu";
    const bool euston_gelu=scope=="euston_gelu"||euston_gelu_trace;
    const bool nexus_gelu=scope=="nexus_gelu";
    const bool blockformer_gelu=scope=="blockformer_gelu";
    const bool trace_gelu=arion_gelu||euston_gelu_trace||moai_gelu||nexus_gelu||blockformer_gelu;
    const bool moai_av=scope=="moai_av", moai_qk=scope=="moai_qk", arion_av=scope=="arion_av", arion_qk=scope=="arion_qk";
    const bool nexus_qk=scope=="nexus_qk",nexus_av=scope=="nexus_av";
    const bool blockformer_qk=scope=="blockformer_qk",blockformer_av=scope=="blockformer_av";
    const bool blockformer_attention=blockformer_qk||blockformer_av;
    const bool av=euston_av||moai_av||arion_av||nexus_av||blockformer_av;
    const bool qk=euston_qk||moai_qk||arion_qk||nexus_qk||blockformer_qk;
    const bool head_slice=av||qk||moai_softmax||euston_softmax||arion_softmax||nexus_softmax||blockformer_softmax;
    const auto separator=scope.find('_');
    const auto method=scope.substr(0,separator);
    const auto stage=separator==std::string::npos?std::string{}:scope.substr(separator+1);
    const bool column_method=method=="moai"||method=="arion";
    const bool recurrent_qkv=scope=="euston_qkv_recurrent"||scope=="nexus_qkv_recurrent";
    const bool euston_iwmm=scope=="euston_initial_qkv";
    const bool nexus_projection_stage=method=="nexus"&&(recurrent_qkv||stage=="wo"||stage=="up"||stage=="down");
    const bool blockformer_projection=method=="blockformer"&&(stage=="qkv"||stage=="wo"||stage=="up"||stage=="down");
    const bool projection=blockformer_projection||nexus_projection_stage||(column_method&&stage=="qkv")||recurrent_qkv||
        ((column_method||method=="euston")&&(stage=="wo"||stage=="up"||stage=="down"));
    const bool norm=(column_method||method=="euston"||method=="nexus"||method=="blockformer")&&(stage=="ln1"||stage=="ln2");
    const bool blockformer_norm=norm&&method=="blockformer";
    const bool nexus_norm=norm&&method=="nexus";
    const bool euston_norm=norm&&method=="euston";
    const bool residual=(column_method||method=="euston"||method=="nexus"||method=="blockformer")&&
        (stage=="attention_residual"||stage=="ffn_residual");
    const bool nexus_online=scope=="nexus_input_projection_tile";
    const bool nexus=scope=="nexus_offline_pcmm"||nexus_online;
    const bool stip_regroup=scope=="stip_amcp_regroup";
    const bool diagnostic=(moai_softmax&&std::getenv("CIPHEROPBENCH_MOAI_BATCH_DIAGNOSTIC"))||
                          (arion_softmax&&std::getenv("CIPHEROPBENCH_ARION_STAGE_DIAGNOSTIC"))||
                          (nexus_softmax&&std::getenv("CIPHEROPBENCH_NEXUS_SOFTMAX_DIAGNOSTIC"));
    bool stip_attention=scope=="stip_l2"||scope=="stip_gk"||scope=="stip_av";
    bool activation=trace_gelu||euston_gelu||scope=="stip_relu";
    const int feature_offset=argc==10?std::stoi(argv[9]):0;
    TORCH_CHECK(head_slice||activation||stip_attention||nexus||stip_regroup||projection||norm||residual||euston_iwmm,"Unknown batch scope");
    const bool blockformer_single=method=="blockformer"&&active==1;
    const int logical_samples=active;
    if(blockformer_single)active=PublishedOperators::block_batch_size;
    TORCH_CHECK(active>=2&&active<=capacity&&warmup>=0&&runs>0&&limit>0&&limit<=(activation?3072:12),
                "Invalid batch geometry or run count");
    TORCH_CHECK(!stip_attention||(active==32&&limit<=2),"STIP AMCP requires 32 samples and up to two head groups");
    TORCH_CHECK(!euston_softmax||active==capacity,"Euston Softmax requires 256 active samples");
    TORCH_CHECK(!nexus_qk||(active==32&&limit==2&&feature_offset==0),"NEXUS QK requires 32 samples and all 12 heads");
    TORCH_CHECK(!(nexus_av||nexus_softmax)||active==32,"NEXUS attention requires a 32-sample batch");
    TORCH_CHECK(!(blockformer_attention||blockformer_softmax)||active==PublishedOperators::block_batch_size,
                "BlockFormer attention requires four independent samples");
    TORCH_CHECK(!arion_softmax||(active==capacity&&std::getenv("CIPHEROPBENCH_ARION_REFRESH_PROFILE")),
                "ARION Softmax requires 256 active samples and a public refresh profile");
    TORCH_CHECK(!trace_gelu||active==(blockformer_gelu?PublishedOperators::block_batch_size:nexus_gelu?32:capacity),
                "FFN trace GELU requires a full native batch");
    TORCH_CHECK(!nexus_gelu||(limit%tokens==0&&feature_offset%tokens==0),"NEXUS GELU requires full feature tiles");
    TORCH_CHECK(!blockformer_gelu||(limit==3072&&feature_offset==0),"BlockFormer GELU requires all FFN blocks");
    TORCH_CHECK(!stip_regroup||(active==32&&limit==1&&warmup==0&&runs==1),"STIP AMCP regrouping requires one measured 32-sample transform");
    const int projection_batch=method=="nexus"?32:method=="blockformer"?PublishedOperators::block_batch_size:capacity;
    TORCH_CHECK(!(projection||norm||residual)||(active==projection_batch&&limit==1),"Operator requires a full native batch");
    const int max_offset=activation?3072-limit:head_slice?12-limit:nexus_online?35:0;
    TORCH_CHECK(feature_offset>=0&&feature_offset<=max_offset,"Invalid operator slice");
    const auto preparation_start=std::chrono::steady_clock::now();
    torch::set_num_threads(2);
    torch::NoGradGuard guard;
    std::ifstream list(argv[6]);
    std::vector<Tensor> hidden, masks;
    std::set<std::filesystem::path> paths;
    Fixture fixture;
    for (int b=0; b<logical_samples; ++b) {
      std::string line;
      TORCH_CHECK(std::getline(list,line)&&!line.empty(),"Missing fixture");
      auto path=std::filesystem::canonical(line);
      TORCH_CHECK(paths.insert(path).second,"Duplicate fixture");
      auto row=load_fixture(path,b==0?1:0);
      hidden.push_back(row.hidden);
      masks.push_back(row.mask[0].to(torch::kBool));
      if (b==0) fixture=std::move(row);
    }
    if(blockformer_single) {
      for(int b=logical_samples;b<active;++b) {
        hidden.push_back(torch::zeros_like(hidden[0]));
        masks.push_back(torch::zeros_like(masks[0]));
      }
    }
    const auto& w=fixture.layers[0];
    if(euston_iwmm){
      TORCH_CHECK((active==32||active==256)&&warmup==0&&runs==1&&limit==1,"Euston IWMM requires one measured 32- or 256-sample pass");
      return euston_initial_qkv(torch::stack(hidden),w,argv[5]);
    }
    if(nexus){
      TORCH_CHECK(active==32&&limit==1,"NEXUS native offline scope requires a batch of 32 and a 64-column projection");
      const auto& weights=feature_offset<12?w.q:feature_offset<24?w.key:w.v;
      return nexus_projection(weights,warmup,runs,argv[5],nexus_online?torch::stack(hidden):Tensor{},feature_offset);
    }
    auto x=torch::stack(hidden), valid=torch::stack(masks);
    std::vector<int> steps;
    if (av||qk) {
      steps={16384};
      for (int i=capacity; i<16384; i*=2) {steps.push_back(i);steps.push_back(-i);}
    }
    if (projection||residual) steps={0};
    if(blockformer_projection){
      for(int i=1;i<64;i*=2)steps.push_back(i);
      for(int i=1;i<64;++i)steps.push_back(-i);
    }
    if(blockformer_attention){
      steps.clear();
      for(int i=1;i<8192;i*=2){steps.push_back(i);steps.push_back(-i);}
    }
    if(blockformer_softmax||blockformer_gelu||blockformer_norm)steps={0};
    if(blockformer_norm)for(int i=1;i<64;i*=2){steps.push_back(i);steps.push_back(-i);}
    if(nexus_projection_stage){
      for(int i=1;i<16;++i)steps.push_back(i);
      // The standard batched rotation decomposes baby steps into signed powers of two.
      for(int i=1;i<16;i*=2)steps.push_back(-i);
      for(int i=-128;i<128;i+=16)if(i)steps.push_back(i);
    }
    if(nexus_av)steps.clear();
    const auto softmax_profiles=euston_softmax||nexus_softmax?
        batch_softmax_profiles(argv[7],nexus_softmax):std::array<EustonSoftmaxProfile,12>{};
    const auto ffn_profile=euston_norm||euston_gelu_trace?euston_ffn_profile(argv[7]):EustonFfnProfile{};
    const auto runtime_start=std::chrono::steady_clock::now();
    Runtime runtime(!(av||qk||projection||residual||method=="blockformer"),steps);
    const auto runtime_ready=std::chrono::steady_clock::now();
    if (moai_softmax) runtime.calibration=calibration::load(argv[7],"moai");
    const bool arion=arion_av||arion_qk||arion_gelu||arion_softmax||((projection||norm||residual)&&method=="arion");
    ColumnOperators columns(runtime,arion?"arion":"moai",arion&&(norm||arion_gelu||arion_softmax)?argv[7]:"");
    LiteratureOperators euston_ops(runtime,"euston",true);
    Timing timing{warmup,std::vector<double>(runs)};
    if(blockformer_single)timing.logical_samples=logical_samples;
    int parts=0;
    double activation_max_approx_error=0.;
    const int norm_input_levels=blockformer_norm?(stage=="ln1"?11:5):nexus_norm?17:euston_norm?16:arion?6:20;
    const double norm_divisor=nexus_norm?(stage=="ln1"?1.:16.):(stage=="ln1"?ffn_profile.ln1:ffn_profile.ln2);
    const int nexus_levels=nexus_projection_stage?(recurrent_qkv?21:stage=="up"?17:2):0;
    if (projection) {
      std::optional<PublishedOperators> block_ops;
      if(blockformer_projection)block_ops.emplace(runtime,"blockformer");
      measure_column_projection(recurrent_qkv?"qkv":stage,runtime,columns,x,valid,w,timing,parts,
                                block_ops?&*block_ops:nullptr,nexus_levels);
    } else if (norm) {
      measure_column_norm(stage=="ln1"?1:2,norm_input_levels,runtime,columns,x,valid,w,timing,
                          euston_norm?&euston_ops:nullptr,norm_divisor,method);
      if(timing.validated) parts=768;
    } else if (residual) {
      measure_column_residual(stage=="attention_residual"?1:2,runtime,columns,x,valid,w,timing,method);
      if(timing.validated) parts=768;
    } else if (stip_regroup) {
      ResearchOperators stip_ops(runtime,"stip","",true);
      auto values=calibration::read_reference(argv[7],size_t(active)*tokens*768);
      auto input=torch::from_blob(values.data(),{active,tokens,768},torch::kDouble);
      auto encrypted=columns.encrypt_batch(input,k::chain_idx(7));
      std::vector<Tensor> expected;
      auto groups=input.reshape({active,tokens,128,6});
      for(int r=0;r<6;++r)expected.push_back(torch::roll(groups,-r,3).permute({3,0,1,2}).reshape({192,tokens,128}));
      timing.measure(runtime,[&]{return stip_ops.batch_stip_regroup(encrypted);},
                     [&](const auto& out){std::vector<Tensor> decoded;
                        for(const auto& group:out)decoded.push_back(columns.decrypt_batch(group,192));
                        return torch::stack(decoded);},torch::stack(expected));
      parts=6;
    } else if (activation) {
      std::optional<ResearchOperators> stip_ops;
      if (!trace_gelu&&!euston_gelu) stip_ops.emplace(runtime,"stip","",true);
      auto pre=trace_gelu?ffn_trace_preactivation(x,valid,w,feature_offset,limit):
          torch::matmul(x,w.up.weight.slice(1,feature_offset,feature_offset+limit))+
          w.up.bias.slice(0,feature_offset,feature_offset+limit);
      if(trace_gelu)std::cerr<<"FFN untimed setup: fixtures_s="
          <<std::chrono::duration<double>(runtime_start-preparation_start).count()<<" runtime_s="
          <<std::chrono::duration<double>(runtime_ready-runtime_start).count()<<" preactivation_s="
          <<std::chrono::duration<double>(std::chrono::steady_clock::now()-runtime_ready).count()<<'\n';
      std::vector<double> slot_mask;
      if(blockformer_gelu){
        PublishedOperators ops(runtime,"blockformer");
        const double linear=std::sqrt(2.*std::numbers::pi),bias=1.;
        auto encrypted=ops.encrypt_block_batch(pre,7);
        auto expected=pre.square()+linear*pre+bias;
        if(blockformer_single) {
          timing.exact_reference=torch::gelu(pre);
          timing.error_mask=valid.unsqueeze(-1);
        }
        CipherList working;
        timing.measure(runtime,[&]{return ops.batch_block_gelu(std::move(working),linear,bias);},
            [&](const CipherList& out){return ops.decrypt_block_batch(out);},expected,[&]{working=encrypted;});
        if(timing.validated)parts=limit;
      } else if(nexus_gelu) {
        TORCH_CHECK(pre.abs().max().item<double>()+4.<=nexus_batch::gelu_sign_divisor,
                    "NEXUS GELU input exceeds its public sign bound");
        auto expected=nexus_gelu_reference(pre);
        activation_max_approx_error=(expected-torch::gelu(pre)).abs().max().item<double>();
        for(int sample=0;sample<active&&timing.validated;sample+=2){
          auto encrypted=encrypt_nexus_pair(runtime,pre.narrow(0,sample,2),16);
          for(int col=0;col<limit&&timing.validated;col+=tokens){
            timing.measure(runtime,[&]{return CipherList{nexus_batch::gelu(runtime,encrypted[col/tokens])};},
                [&](const CipherList& out){return runtime.decrypt_slots(out[0]);},
                expected.narrow(0,sample,2).narrow(2,col,tokens));
          }
        }
        if(timing.validated)parts=limit;
      } else if(arion_gelu) {
        const auto [low,high]=columns.batch_arion_gelu_domain();
        TORCH_CHECK(pre.min().item<double>()>=low&&pre.max().item<double>()<=high,
                    "ARION GELU activation exceeds its public polynomial range");
        slot_mask=interleaved_mask(valid);
      } else if(moai_gelu) {
        TORCH_CHECK(pre.abs().max().item<double>()<=ColumnOperators::moai_batch_gelu_bound,
                    "MOAI GELU activation exceeds its public extended interval");
        slot_mask=interleaved_mask(valid);
      } else {
        TORCH_CHECK(pre.abs().max().item<double>()<=ffn_profile.gelu,"FFN activation exceeds its public sign interval");
      }
      for (int feature=0; !(nexus_gelu||blockformer_gelu)&&feature<limit&&timing.validated; ++feature) {
        auto lane=pre.select(2,feature).transpose(0,1).contiguous();
        auto input=lane.accessor<double,2>();
        k::FlatVec packed(slots);
        // FFN columns retain the same sample lanes through FC1, activation and FC2.
        for (int t=0; t<tokens; ++t) for (int b=0; b<active; ++b)
          packed[t*capacity+b]=make_cuDoubleComplex(input[t][b],0.);
        auto ciphertext=runtime.encrypt(std::move(packed),k::chain_idx(arion_gelu||moai_gelu?6:euston_gelu_trace?15:euston_gelu?16:14));
        auto expected=arion_gelu?torch::gelu(lane):euston_gelu?euston_reference(lane,ffn_profile.gelu):
            moai_gelu?columns.moai_batch_gelu_reference(lane).masked_fill(~valid.transpose(0,1),0.):
            lane*(series(series(lane/16.,cipheropbench::detail::STIP_RELU_0),cipheropbench::detail::STIP_RELU_1)+.5);
        if(moai_gelu)activation_max_approx_error=std::max(activation_max_approx_error,
            (expected-torch::gelu(lane)).abs().max().item<double>());
        timing.measure(runtime,[&]{return CipherList{arion_gelu?columns.batch_arion_gelu(ciphertext,slot_mask):
                           moai_gelu?columns.batch_moai_gelu(ciphertext,slot_mask):
                           euston_gelu?euston_ops.batch_gelu(ciphertext,ffn_profile.gelu):stip_ops->batch_stip_relu(ciphertext)};},
                       [&](const CipherList& out){return runtime.decrypt_slots(out[0]).reshape({tokens,capacity}).narrow(1,0,active);},expected);
        ++parts;
        if (parts%32==0||!timing.validated) std::cerr<<"Activation ciphertexts checked: "<<parts<<" max_error="<<timing.max_error<<'\n';
      }
    } else if(blockformer_attention||blockformer_softmax) {
      PublishedOperators ops(runtime,"blockformer");
      auto query=project_heads(x,w.q),key=project_heads(x,w.key);
      auto values=blockformer_av?project_heads(x,w.v):Tensor{};
      auto keep=valid.unsqueeze(-1);
      for(int head=feature_offset;head<feature_offset+limit&&timing.validated;++head){
        auto a=query.select(1,head)*keep,b=key.select(1,head)*keep;
        auto scores=torch::matmul(a,b.transpose(1,2))/8.;
        auto decode=[&](const CipherList& out){return ops.decrypt_block_batch(out);};
        if(blockformer_single)timing.error_mask=valid.unsqueeze(-1)&valid.unsqueeze(1);
        if(blockformer_qk){
          auto encrypted_q=ops.encrypt_block_batch(a),encrypted_k=ops.encrypt_block_batch(b);
          timing.measure(runtime,[&]{return ops.batch_block_qk(encrypted_q[0],encrypted_k[0]);},decode,scores);
        }else{
          auto pair=valid.unsqueeze(1)&valid.unsqueeze(2);
          auto quartic=(scores+PublishedOperators::block_softmax_shift).pow(4).masked_fill(~pair,0.);
          constexpr auto divisor=PublishedOperators::block_softmax_divisor;
          TORCH_CHECK(quartic.max().item<double>()<=divisor,"BlockFormer quartic exceeds the public normalizer");
          if(blockformer_softmax){
            auto clear=scores.masked_fill(~pair,0.);
            auto encrypted=ops.encrypt_block_batch(clear,8);
            k::FlatVecArray shift;
            auto offsets=PublishedOperators::block_softmax_shift*pair.to(torch::kDouble);
            for(int col=0;col<tokens;col+=64)shift.push_back(k::vector_from_tensor(offsets.narrow(2,col,64)));
            CipherList working;
            if(blockformer_single)
              timing.exact_reference=torch::softmax(scores.masked_fill(~valid.unsqueeze(1),-100.),-1).masked_fill(~pair,0.);
            timing.measure(runtime,[&]{return ops.batch_block_softmax(std::move(working),shift);},
                           decode,quartic/divisor,[&]{working=encrypted;},block_softmax_validation_atol,0.);
            if(!timing.validated)
              std::cerr<<"BlockFormer Softmax input max_error="<<(decode(encrypted)-clear).abs().max().item<double>()<<'\n';
          }else{
            auto p=quartic/divisor;
            auto v=values.select(1,head)*keep;
            auto encrypted_p=ops.encrypt_block_batch(p),encrypted_v=ops.encrypt_block_batch(v);
            if(blockformer_single) {
              auto exact=torch::softmax(scores.masked_fill(~valid.unsqueeze(1),-100.),-1).masked_fill(~pair,0.);
              timing.exact_reference=torch::matmul(exact,v);
              timing.error_mask=valid.unsqueeze(-1);
            }
            timing.measure(runtime,[&]{return CipherList{ops.batch_block_av(encrypted_p,encrypted_v[0])};},
                           decode,torch::matmul(p,v));
          }
        }
        if(timing.validated)++parts;
      }
    } else if(nexus_softmax) {
      auto query=project_heads(x,w.q),key=project_heads(x,w.key);
      auto valid_pairs=valid.unsqueeze(1)&valid.unsqueeze(2);
      auto packed_mask=diagonals(valid_pairs.to(torch::kDouble));
      for(int head=feature_offset;head<feature_offset+limit&&timing.validated;++head){
        const auto& profile=softmax_profiles[head];
        auto scores=torch::matmul(query.select(1,head),key.select(1,head).transpose(1,2))/8.;
        auto z=(scores-profile.shift).masked_fill(~valid_pairs,0.)/256.;
        TORCH_CHECK(z.min().item<double>()>-1.&&z.max().item<double>()<=0.,
                    "NEXUS exponent input exceeds the public range");
        auto e=(1.+z).pow(256).masked_fill(~valid_pairs,0.)/profile.divisor;
        auto sum=e.sum(-1,true).masked_fill(~valid.unsqueeze(-1),1.);
        TORCH_CHECK(sum.min().item<double>()>0.&&sum.max().item<double>()<2.,
                    "NEXUS denominator outside the Goldschmidt interval");
        const auto repeats=std::exp2(profile.inverse_iterations+1);
        auto expected=diagonals(e*(1.-(1.-sum).pow(repeats))/sum);
        auto packed=diagonals(scores.masked_fill(~valid_pairs,0.));
        for(int sample=0;sample<active&&timing.validated;sample+=2){
          auto encrypted=runtime.encrypt(k::flatten_pack(packed[sample],packed[sample+1]),k::chain_idx(19));
          auto mask=k::vector_from_tensor<double>(packed_mask.narrow(0,sample,2));
          timing.measure(runtime,[&]{return CipherList{nexus_batch::softmax(runtime,encrypted,mask,
                                              profile.shift,profile.divisor,profile.inverse_iterations)};},
              [&](const CipherList& out){return runtime.decrypt_slots(out[0]);},expected.narrow(0,sample,2));
        }
        if(timing.validated)++parts;
      }
    } else if(nexus_av) {
      TransformerOperators ops(runtime);
      auto query=project_heads(x,w.q),key=project_heads(x,w.key),values=project_heads(x,w.v);
      for(int head=feature_offset;head<feature_offset+limit&&timing.validated;++head){
        auto scores=torch::matmul(query.select(1,head),key.select(1,head).transpose(1,2))/8.;
        auto probabilities=torch::softmax(scores.masked_fill(~valid.unsqueeze(1),-100.),-1)*valid.unsqueeze(2);
        auto value=values.select(1,head)*valid.unsqueeze(2);
        auto expected=torch::matmul(probabilities,value);
        for(int sample=0;sample<active&&timing.validated;sample+=2){
          auto p=runtime.encrypt(k::flatten_pack(ops.diagonalize(probabilities[sample]),
                                                  ops.diagonalize(probabilities[sample+1])),k::chain_idx(3));
          auto v=runtime.encrypt(k::flatten_pack(torch::cat({value[sample],value[sample+1]},1)),k::chain_idx(20));
          timing.measure(runtime,[&]{return CipherList{ops.av(p,v)};},
                         [&](const CipherList& out){return ops.unpack_av(out[0]);},expected.narrow(0,sample,2));
        }
        if(timing.validated)++parts;
      }
    } else if (nexus_qk) {
      auto query=project_heads(x,w.q),key=project_heads(x,w.key);
      for(int start=0;start<12&&timing.validated;start+=8){
        const int heads=std::min(8,12-start);
        const auto keep=valid.repeat({heads,1});
        auto pack_heads=[&](const Tensor& tensor){
          return tensor.narrow(1,start,heads).transpose(0,1).reshape({heads*active,tokens,64})*keep.unsqueeze(-1);
        };
        auto q=pack_heads(query),keys=pack_heads(key);
        auto cq=columns.encrypt_batch(q),ck=columns.encrypt_batch(keys);
        cipheropbench::detail::Math m{runtime};m.prime_scale=true;m.column_token_stride=capacity;
        timing.measure(runtime,[&]{return cipheropbench::detail::column_qk(m,cq,ck,false);},
            [&](const CipherList& out){return columns.decrypt_batch(out,heads*active);},
            diagonals(torch::matmul(q,keys.transpose(1,2))/8.));
        if(timing.validated)parts+=heads;
      }
    } else if (stip_attention) {
      ResearchOperators stip_ops(runtime,"stip","",true);
      const auto keep=valid.repeat({6,1});
      const auto pair=keep.unsqueeze(1)&keep.unsqueeze(2);
      for (int group=0; group<limit && timing.validated; ++group) {
        auto project=[&](const LinearWeights& weights){
          const int lo=group*6*64, hi=lo+6*64;
          auto values=torch::matmul(x,weights.weight.slice(1,lo,hi))+weights.bias.slice(0,lo,hi);
          return values.reshape({32,tokens,6,64}).permute({2,0,1,3}).reshape({192,tokens,64});
        };
        auto q=project(w.q), key=project(w.key);
        auto decode=[&](const CipherList& out){return columns.decrypt_batch(out,192);};
        auto l2=q.square().sum(-1,true)+key.square().sum(-1).unsqueeze(1)-2*torch::matmul(q,key.transpose(1,2));
        if(scope=="stip_l2"){
          auto encrypted_q=columns.encrypt_batch(q), encrypted_key=columns.encrypt_batch(key);
          timing.measure(runtime,[&]{return stip_ops.batch_stip_l2(encrypted_q,encrypted_key);},decode,diagonals(l2));
        }else{
          auto exponent=-l2.masked_fill(~pair,0.)/(16.*32);
          auto p=polynomial(exponent,cipheropbench::literature::EUSTON_EXP).pow(32).masked_fill(~pair,0.);
          if(scope=="stip_gk"){
            auto encrypted=columns.encrypt_batch(diagonals(l2));
            timing.measure(runtime,[&]{return stip_ops.batch_stip_gk(encrypted,valid);},decode,diagonals(p));
          }else{
            auto v=project(w.v);
            auto encrypted_p=columns.encrypt_batch(diagonals(p)), encrypted_v=columns.encrypt_batch(v);
            timing.measure(runtime,[&]{return stip_ops.batch_stip_av(encrypted_p,encrypted_v);},decode,torch::matmul(p,v));
          }
        }
        ++parts;
        std::cerr<<"AMCP head groups checked: "<<parts<<'/'<<limit<<" passed="<<timing.validated<<" max_error="<<timing.max_error<<'\n';
      }
    } else {
      auto q=project_heads(x,w.q), key=project_heads(x,w.key);
      auto values=av||arion_softmax?project_heads(x,w.v):Tensor{};
      auto pair=valid.unsqueeze(1)&valid.unsqueeze(2);
      for (int h=feature_offset; h<feature_offset+limit && timing.validated; ++h) {
        auto query=q.select(1,h), keys=key.select(1,h);
        auto scores=torch::matmul(query,keys.transpose(1,2))/8.;
        auto decode=[&](const CipherList& out){return columns.decrypt_batch(out,active);};
        if (qk) {
          auto encrypted_q=columns.encrypt_batch(query*valid.unsqueeze(-1));
          auto encrypted_k=columns.encrypt_batch(keys*valid.unsqueeze(-1));
          cipheropbench::detail::Math m{runtime};m.prime_scale=true;m.column_token_stride=capacity;
          timing.measure(runtime,[&]{
            if(arion_qk)return cipheropbench::detail::column_qk(m,encrypted_q,encrypted_k,true);
            if(euston_qk)return euston_ops.batch_qk(encrypted_q,encrypted_k);
            return columns.batch_qk(encrypted_q,encrypted_k);
          },decode,diagonals(scores.masked_fill(~pair,0.),arion_qk));
        } else if (moai_softmax) {
          auto p=columns.encrypt_batch(diagonals(scores.masked_fill(~pair,0.)));
          timing.measure(runtime,[&]{return columns.batch_softmax(p,valid);},
                         decode,softmax_reference(scores,valid,runtime.calibration.layers[0]));
        } else if (euston_softmax) {
          const auto& profile=softmax_profiles[h];
          auto expected=euston_softmax_reference(scores,valid,profile);
          auto encrypted=columns.encrypt_batch(diagonals(scores.masked_fill(~pair,0.)),k::chain_idx(6));
          CipherList working;
          timing.measure(runtime,[&]{return euston_ops.batch_softmax(std::move(working),valid,profile);},
                         decode,expected,[&]{working=encrypted;});
        } else if (arion_softmax) {
          TORCH_CHECK(scores.abs().max().item<double>()<=ColumnOperators::arion_batch_score_bound,
                      "ARION scores exceed the public refresh bound");
          auto reference=columns.arion_softmax_reference(scores,values.select(1,h)*valid.unsqueeze(-1),valid);
          auto expected=torch::cat({diagonals(reference.exponentials,true),reference.normalized_av},-1);
          auto encrypted=columns.encrypt_batch(diagonals(scores.masked_fill(~pair,0.),true),k::chain_idx(4));
          auto encrypted_av=columns.encrypt_batch(reference.unnormalized_av,k::chain_idx(6));
          CipherList working,working_av;
          timing.measure(runtime,[&]{
            return columns.batch_arion_softmax(std::move(working),std::move(working_av),valid,h);
          },[&](const CipherList& out){
            auto result=decode(out);
            if(std::getenv("CIPHEROPBENCH_ARION_SOFTMAX_DIAGNOSTIC")){
              auto error=(result-expected).abs();
              for(const auto& [start,width]:std::array<std::pair<int,int>,2>{{{0,128},{128,64}}}){
                auto part=error.narrow(-1,start,width);
                std::cerr<<"ARION output "<<start<<" max_error="<<part.max().item<double>()
                         <<" mae="<<part.mean().item<double>()<<'\n';
              }
            }
            return result;
          },expected,[&]{working=encrypted;working_av=encrypted_av;});
        } else {
          Tensor attention;
          if (arion_av) {
            const auto count=valid.sum(1,true).clamp_min(1).unsqueeze(-1);
            const auto mean=scores.masked_fill(~pair,0.).sum(-1,true)/count;
            const auto variance=(scores-mean).square().masked_fill(~pair,0.).sum(-1,true)/count;
            // The ARION AV product consumes exponentials before the row reciprocal.
            attention=(scores-mean-ColumnProfile{}.statmax*(1.25+.1*variance)).exp().masked_fill(~pair,0.);
          } else {
            attention=torch::softmax(scores.masked_fill(~valid.unsqueeze(1),-100.),-1)*pair;
          }
          auto p=columns.encrypt_batch(diagonals(attention,arion_av));
          auto v=values.select(1,h)*valid.unsqueeze(-1);
          auto encrypted=columns.encrypt_batch(v);
          timing.measure(runtime,[&]{return euston_av?euston_ops.batch_av(p,encrypted):columns.batch_av(p,encrypted);},
                         decode,torch::matmul(attention,v));
        }
        ++parts;
        std::cerr<<"Heads checked: "<<parts<<'/'<<limit<<" passed="<<timing.validated<<" max_error="<<timing.max_error<<'\n';
      }
    }
    std::ofstream out(argv[5],std::ios::app);
    TORCH_CHECK(out.good(),"Cannot open output");
    const char* refresh_gain=std::getenv("CIPHEROPBENCH_LEGACY_PRECISION_GAIN");
    const auto method_name=method=="moai"?"MOAI":method=="euston"?"EUSTON":method=="blockformer"?"BLOCKFORMER":method=="nexus"?"NEXUS":arion?"ARION":"STIP";
    const bool columns_only=projection||norm||residual;
    const bool nexus_paired=nexus_projection_stage||nexus_av||nexus_norm||nexus_gelu||nexus_softmax||(method=="nexus"&&residual);
    const char* profile_name="published_domain_library";
    if(blockformer_qk)profile_name="published_blockformer_ssbmm_alpha64";
    else if(blockformer_av)profile_name="published_blockformer_srbmm_alpha64";
    else if(blockformer_softmax)profile_name="published_blockformer_quartic";
    else if(blockformer_gelu)profile_name="published_blockformer_folded_quadratic";
    else if(blockformer_norm)profile_name="published_blockformer_affine_center";
    else if(method=="blockformer"&&residual)profile_name="published_blockformer_residual";
    else if(nexus_gelu)profile_name="nexus_piecewise_puma_3_6";
    else if(nexus_softmax)profile_name="nexus_exp8_goldschmidt";
    else if(nexus_norm)profile_name="nexus_row_newton_goldschmidt";
    else if(nexus_paired)profile_name=residual?"nexus_row_residual":"nexus_row_bsgs";
    else if(blockformer_projection)profile_name="published_blockformer_pcbmm_alpha64";
    else if(residual)profile_name="library_column_residual";
    else if(norm)profile_name=euston_norm?"published_euston_cubic_root":"library_column_layernorm";
    else if(projection)profile_name="library_column_cpmm";
    else if(moai_softmax)profile_name="sst2_train_calibrated_goldschmidt";
    else if(euston_softmax)profile_name="published_diag_exp7_goldschmidt";
    else if(arion_softmax)profile_name="published_arion_exp63_deferred_reciprocal";
    else if(av)profile_name=arion_av?"library_diagonal_av_before_reciprocal":"library_diagonal_av";
    else if(qk)profile_name="library_diagonal_qk";
    else if(arion_gelu)profile_name="published_arion_chebyshev255";
    else if(moai_gelu)profile_name="published_moai_release_gate_cubic_dep";
    else if(euston_gelu)profile_name=euston_gelu_trace?"published_trace_depth4_sign":"published_domain_depth4_sign";
    else if(activation)profile_name="published_domain_exact_gain_relu";
    const char* layout=nexus_paired?"nexus_paired_128x128_blocks":
        method=="blockformer"?"sample_major_64_feature_blocks":
        nexus_qk?"nexus_column_diagonals_sample_head_groups":
        columns_only?"token_sample_interleaved_columns":
        activation?"ffn_feature_columns_token_sample_interleaved":
        stip_regroup?"stip_amcp_regroup_6_by_32":stip_attention?"stip_amcp_6_heads":
        arion_av||arion_qk||arion_softmax?"giant_shifted_diagonals_token_sample_interleaved":
                                        "attention_diagonals_token_sample_interleaved";
    out<<std::setprecision(14)<<"{\"implementation\":\"cipheropbench_common_library\",\"method\":\""
       <<method_name<<"\",\"scope\":\""<<scope
       <<"\",\"active_samples\":"<<logical_samples<<",\"sequence_length\":"<<tokens<<",\"N\":"<<k::N
       <<",\"reference_kind\":\""<<((columns_only&&!(norm&&method!="arion"))||arion_gelu||nexus_qk||nexus_av||blockformer_qk?"exact_transformer":"noiseless_cipheropbench_method_circuit")
       <<"\",\"input\":\"distinct_model_traces\""
       <<",\"profile\":\""<<profile_name<<"\",\"layout\":\""<<layout
       <<"\",\"refresh_profile\":\""<<(std::getenv("CIPHEROPBENCH_LEGACY_PRECISION_REFRESH")?"legacy_precision":"legacy_default")
       <<"\",\"refresh_gain\":"<<(refresh_gain?std::stoi(refresh_gain):256)
       <<",\"secret_key_bridge_in_timing\":false,\"warmup\":"<<warmup<<",\"runs\":"<<runs
       <<",\""<<(columns_only?"output_features":activation?"ffn_features":stip_regroup?"packing_group_states":"attention_heads")<<"\":"<<(columns_only?parts:nexus_qk?12:stip_attention||stip_regroup?6*limit:limit)
       <<",\""<<(head_slice?"head_offset":"feature_offset")<<"\":"<<feature_offset
       <<",\"parts_checked\":"<<parts<<",\"max_abs_error\":"<<timing.max_error<<",\"max_mae\":"<<timing.max_mae
       <<",\"status\":\""<<(diagnostic?"diagnostic_only":timing.validated?"validated":"numerical_mismatch")<<"\"";
    if(norm||euston_softmax||residual||trace_gelu||blockformer_softmax)
      out<<",\"input_levels\":"<<(norm?norm_input_levels:nexus_gelu?16:euston_gelu_trace?15:
                                 blockformer_gelu?7:blockformer_softmax?8:nexus_paired?1:6);
    if(nexus_projection_stage||nexus_av||nexus_softmax)out<<",\"input_levels\":"<<(nexus_av?3:nexus_softmax?19:nexus_levels);
    if(nexus_av||(nexus_projection_stage&&nexus_levels>k::L+1))
      out<<",\"bootstrap_prime_scaling\":\"preserve_common_scale\"";
    if(nexus_av)out<<",\"value_input_levels\":20";
    if(nexus_norm)out<<",\"variance_refresh_bound\":"<<norm_divisor
                     <<",\"variance_refresh_arithmetic\":\"single_bootstrap\"";
    if(nexus_paired)out<<",\"physical_group_size\":2,\"measured_group_count\":"<<active/2;
    if(blockformer_attention)out<<",\"input_levels\":8,\"rotation_key_schedule\":\"signed_binary_composition\"";
    if(blockformer_gelu)out<<",\"gelu_linear_coefficient\":"<<std::sqrt(2.*std::numbers::pi)
                          <<",\"gelu_bias\":1,\"leading_coefficient_folded\":true";
    if(blockformer_softmax||blockformer_av)
      out<<",\"softmax_shift\":"<<PublishedOperators::block_softmax_shift<<",\"softmax_power\":4"
         <<",\"softmax_divisor\":"<<PublishedOperators::block_softmax_divisor;
    if(blockformer_softmax)out<<",\"validation_error_units\":\"normalized_quartic\""
         <<",\"validation_atol\":"<<block_softmax_validation_atol
         <<",\"softmax_working_scale_bits\":"<<PublishedOperators::block_softmax_working_scale_bits;
    if(blockformer_norm)out<<",\"normalization_affine_folded\":true";
    if(trace_gelu) out<<",\"activation_inputs\":\"layer0_ffn_preactivation\"";
    else if(euston_gelu) out<<",\"activation_inputs\":\"embedding_projection\"";
    if(euston_norm) out<<",\"variance_divisor\":"<<norm_divisor;
    if(euston_gelu_trace) out<<",\"gelu_sign_bound\":"<<ffn_profile.gelu;
    if(nexus_gelu)out<<",\"gelu_sign_bound\":"<<nexus_batch::gelu_sign_divisor
                    <<",\"gelu_max_approximation_error\":"<<activation_max_approx_error;
    if(nexus_softmax&&limit==1){
      const auto& profile=softmax_profiles[feature_offset];
      const int exp_gain=nexus_batch::softmax_exponential_gain_bits(profile.inverse_iterations,profile.divisor);
      out<<",\"softmax_shift\":"<<profile.shift<<",\"softmax_divisor\":"<<profile.divisor
         <<",\"softmax_inverse_iterations\":"<<profile.inverse_iterations
         <<",\"softmax_inverse_state\":\"complement\""
         <<",\"softmax_exponential_gain_bits\":"<<exp_gain
         <<",\"softmax_denominator_gain_bits\":"<<nexus_batch::softmax_denominator_gain_bits(exp_gain)
         <<",\"softmax_exponential_data_levels\":"<<k::L+1;
    }
    if(nexus_qk) out<<",\"head_groups\":[8,4]";
    if(moai_gelu) out<<",\"gelu_domain_bound\":"<<ColumnOperators::moai_batch_gelu_bound
                    <<",\"gelu_max_approximation_error\":"<<activation_max_approx_error;
    if(arion_softmax){
      out<<",\"input_levels\":4,\"deferred_av_input_levels\":6,\"score_refresh_bound\":"
         <<ColumnOperators::arion_batch_score_bound;
      if(const auto* value=std::getenv("CIPHEROPBENCH_ARION_EXP_REBASE");value&&std::string(value)=="1")
        out<<",\"exponential_basis_low\":"<<ColumnOperators::arion_batch_exp_low
           <<",\"exponential_basis_high\":"<<ColumnOperators::arion_batch_exp_high;
    }
    if(blockformer_single)
      out<<",\"physical_batch_capacity\":4,\"zero_padded_samples\":3,\"exact_bert_mae\":"
         <<timing.exact_error_sum/timing.exact_error_count
         <<",\"error_scope\":\"active_sample_valid_tokens\",\"numerical_validation_in_timing\":false";
    out<<",\"batch_seconds\":[";
    if (timing.validated&&!diagnostic) for (int i=0; i<runs; ++i) out<<(i?",":"")<<timing.seconds[i];
    out<<"]}\n";
    return timing.validated?0:2;
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
  }
}
