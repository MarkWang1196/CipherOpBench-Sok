#pragma once
#include "cipheropbench/kernels/nexus_utility.cuh"
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cipheropbench::detail {
inline void legacy_refresh_sync(){
  auto error=cudaDeviceSynchronize();
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
// A second bootstrap of the amplified residual reduces the refresh error.
// Both passes and final scale normalization are server-side encrypted work.
inline void legacy_precision_refresh(PhantomCiphertext& input,
    const std::shared_ptr<Bootstrapper>& bootstrapper) {
  static const unsigned gain=[] {
    const char* value=std::getenv("CIPHEROPBENCH_LEGACY_PRECISION_GAIN");
    if(!value)return 256U;
    size_t used=0;
    auto result=std::stoul(value,&used);
    if(used!=std::string(value).size()||result<256||result>4096||(result&(result-1)))
      throw std::invalid_argument("Expected a power-of-two precision gain in [256,4096]");
    return static_cast<unsigned>(result);
  }();
  auto& ckks=*bootstrapper->ckks;
  if(std::abs(std::log2(input.scale())-kernels::logp)>1e-8)
    throw std::invalid_argument("Precision refresh requires the common input scale");
  auto original=input;
  while(original.coeff_modulus_size()>1)ckks.evaluator.mod_switch_to_next_inplace(original);
  auto first=original;kernels::bootstrap(first,bootstrapper,true);legacy_refresh_sync();
  auto reduced=first;
  while(reduced.coeff_modulus_size()>1)ckks.evaluator.mod_switch_to_next_inplace(reduced);
  PhantomCiphertext residual;ckks.evaluator.sub(original,reduced,residual);
  for(unsigned g=1;g<gain;g*=2)ckks.evaluator.add_inplace(residual,residual);
  kernels::bootstrap(residual,bootstrapper,true);legacy_refresh_sync();
  if(first.chain_index()!=residual.chain_index())throw std::runtime_error("Refresh passes returned different levels");
  for(unsigned g=1;g<gain;g*=2)ckks.evaluator.add_inplace(first,first);
  ckks.evaluator.add_inplace(first,residual);
  auto prime=ckks.context->get_context_data(first.chain_index()).parms().coeff_modulus().back().value();
  PhantomPlaintext coefficient;
  ckks.encoder.encode(std::vector<double>(kernels::slot_count,1./gain),first.chain_index(),
                      prime*kernels::scale/first.scale(),coefficient);
  ckks.evaluator.multiply_plain_inplace(first,coefficient);
  ckks.evaluator.rescale_to_next_inplace(first);first.set_scale(kernels::scale);
  legacy_refresh_sync();input=std::move(first);
}
}
