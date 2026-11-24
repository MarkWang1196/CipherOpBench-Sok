#pragma once
#include "ckks_evaluator.cuh"
#include "Bootstrapper.cuh"
#include "cipheropbench/unified_phantom_parameters.cuh"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace cipheropbench::unified_phantom {
inline void refresh_sync(){auto status=cudaDeviceSynchronize();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
inline void refresh_drop(nexus::CKKSEvaluator& evaluator,PhantomCiphertext& x,size_t moduli){
  if(moduli<1||moduli>x.coeff_modulus_size())throw std::invalid_argument("Refresh modulus budget");
  while(x.coeff_modulus_size()>moduli)evaluator.evaluator.mod_switch_to_next_inplace(x);
}
inline PhantomCiphertext refresh_integer(nexus::CKKSEvaluator& evaluator,PhantomCiphertext x,unsigned factor){
  if(!factor)throw std::invalid_argument("Nonzero refresh gain required");
  PhantomCiphertext base=x,result;bool first=true;
  while(factor){if(factor&1){if(first)result=base;else evaluator.evaluator.add_inplace(result,base);first=false;}
    factor>>=1;if(factor)evaluator.evaluator.add_inplace(base,base);}
  return result;
}

// Public two-pass residual correction. If B(x)=x+e1 and
// B(g*(x-B(x)))=g*(x-B(x))+e2, the result is x+e2/g.
// The final real plaintext multiplication also applies a public output mask.
// No private key, decoded value, or client callback is used here.
inline PhantomCiphertext precision_refresh(nexus::CKKSEvaluator& evaluator,
    Bootstrapper& bootstrapper,PhantomCiphertext input,bool real,
    double amplitude=1.,const std::vector<double>& output_mask={}){
  constexpr unsigned gain=precision_refresh_gain;
  if(input.size()!=2||!input.is_ntt_form()||
     std::abs(std::log2(input.scale())-log_scale)>1e-8||
     !std::isfinite(amplitude)||amplitude<=0)
    throw std::invalid_argument("Unified precision refresh input");
  const size_t count=evaluator.slot_count;
  if(!output_mask.empty()&&output_mask.size()!=count)throw std::invalid_argument("Refresh mask shape");
  for(double value:output_mask)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite public refresh mask");
  refresh_drop(evaluator,input,1);auto original=input;
  auto bootstrap=[&](PhantomCiphertext argument){PhantomCiphertext result;bootstrapper.set_final_scale(scale);
    if(real)bootstrapper.bootstrap_real_3(result,argument);else bootstrapper.bootstrap_3(result,argument);
    refresh_sync();return result;};
  auto first=bootstrap(input);auto reduced=first;refresh_drop(evaluator,reduced,1);
  PhantomCiphertext residual;evaluator.evaluator.sub(original,reduced,residual);
  residual=refresh_integer(evaluator,std::move(residual),gain);
  auto correction=bootstrap(std::move(residual));
  if(first.coeff_modulus_size()!=correction.coeff_modulus_size()||
     first.coeff_modulus_size()!=size_t(working_moduli+1))
    throw std::runtime_error("Precision refresh must reserve one normalization prime");
  auto sum=refresh_integer(evaluator,std::move(first),gain);
  evaluator.evaluator.add_inplace(sum,correction);
  std::vector<double> coefficients(count,amplitude/gain);
  if(!output_mask.empty())for(size_t i=0;i<count;++i)coefficients[i]*=output_mask[i];
  double prime=evaluator.context->get_context_data(sum.chain_index()).parms().coeff_modulus().back().value();
  PhantomPlaintext plain;evaluator.encoder.encode(coefficients,sum.chain_index(),prime*scale/sum.scale(),plain);
  evaluator.evaluator.multiply_plain_inplace(sum,plain);evaluator.evaluator.rescale_to_next_inplace(sum);
  if(sum.coeff_modulus_size()!=size_t(working_moduli)||std::abs(sum.scale()/scale-1)>1e-12)
    throw std::runtime_error("Precision refresh output scale/level normalization");
  sum.set_scale(scale);refresh_sync();return sum;
}
}
