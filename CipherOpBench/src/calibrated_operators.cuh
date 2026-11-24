#pragma once
#include "literature_math.cuh"
#include "cipheropbench/calibrated_math.hpp"
namespace cipheropbench::detail {
inline CipherList calibrated_gelu(Runtime& r,CipherList x,int layer){
 Math m{r};CipherList out;
 r.record("calibrated_gelu_with_tail_gates",[&]{for(auto& ct:x)out.push_back(calibration::gelu(m,ct,r.calibration.layers.at(layer),r.calibration.method));});return out;
}
inline Cipher calibrated_softmax(Runtime& r,Cipher x,const k::FlatVec& mask,int layer){
 Math m{r};auto& p=r.calibration.layers.at(layer);Cipher result;
 r.record("calibrated_projective_softmax",[&]{
  Vec valid(slots);for(int i=0;i<slots;++i)valid[i]=cuCreal(mask[i])==0?1.:0.;
  auto y=m.pm(calibration::softmax_base(m,x,p,r.calibration.method),valid);
  auto normalize=[&](Cipher value,double maximum){auto inverse=calibration::scaled_reciprocal(m,m.row_sum(value),maximum);auto out=m.mul(value,inverse);m.integer_scale(out,int64_t(128/maximum));return out;};
  y=normalize(y,128.);int count=r.calibration.method=="euston"?p.euston_softmax_squarings:p.exp_squarings;
  for(int i=0;i<count;++i)y=normalize(m.square(y),1.);result=std::move(y);
 });return result;
}
inline CipherList calibrated_norm(Runtime& r,CipherList x,torch::Tensor weight,torch::Tensor bias,int layer,int kind){
 Math m{r};CipherList out;
 r.record("calibrated_layernorm",[&]{
  auto mean=m.scale(m.hidden_sum(x),1./768);CipherList center,squares;
  for(auto& ct:x){center.push_back(m.sub(ct,mean));squares.push_back(m.square(center.back()));}
  auto variance=m.addc(m.scale(m.hidden_sum(squares),1./768),r.calibration.epsilon);
  auto inverse=calibration::inverse_sqrt(m,variance,r.calibration.layers.at(layer),kind,r.calibration.method);
  auto gamma=k::row_pack_768x1(weight),beta=k::row_pack_768x1(bias);
  for(int i=0;i<3;++i){auto y=m.mul(center[i],inverse);Vec weights(slots);for(int j=0;j<slots;++j)weights[j]=cuCreal(gamma[i][j]);y=m.pm(y,weights);auto b=k::CKKSEncode(beta[i],r.ckks,&y);r.ckks->evaluator.add_plain_inplace(y,b);out.push_back(std::move(y));}
 });return out;
}
inline void calibrated_mask(Runtime& r,CipherList& x,torch::Tensor mask){
 if(!r.calibration.enabled)return;Math m{r};auto valid=mask.diag().to(torch::kDouble).view({128,1}).expand({128,768});auto packed=k::row_pack_128x768(valid);
 r.record("calibrated_padding_projection",[&]{for(int i=0;i<3;++i){Vec v(slots);for(int j=0;j<slots;++j)v[j]=cuCreal(packed[i][j]);x[i]=m.pm(x[i],v);}});
}
}
