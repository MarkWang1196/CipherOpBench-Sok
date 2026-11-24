#pragma once
#include "cipheropbench/runtime.cuh"
#include "cipheropbench/arion_scalar_rns.hpp"
namespace cipheropbench::detail {
static __global__ void scalar_multiply_add(const uint64_t* input,uint64_t* output,const DModulus* moduli,ArionScalarRNS scalar,size_t n,bool add){
 size_t size=2*n*scalar.levels;
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=size_t(blockDim.x)*gridDim.x){
  size_t prime=i/n%scalar.levels;uint64_t q=moduli[prime].value();
  auto product=phantom::arith::multiply_and_reduce_shoup(input[i],scalar.value[prime],scalar.shoup[prime],q);
  output[i]=add?phantom::arith::add_uint64_uint64_mod(output[i],product,q):product;
 }
}
static inline void arion_scalar_accumulate(const PhantomContext& context,const Cipher& input,Cipher& output,const ArionScalarRNS& scalar,bool add){
 TORCH_CHECK(scalar.levels>=1&&scalar.levels<=15&&input.is_ntt_form()&&output.is_ntt_form()&&input.size()==2&&output.size()==2&&input.poly_modulus_degree()==65536&&output.poly_modulus_degree()==65536&&input.chain_index()==output.chain_index()&&input.coeff_modulus_size()==size_t(scalar.levels)&&output.coeff_modulus_size()==size_t(scalar.levels)&&input.parms_id()==output.parms_id()&&input.data()!=output.data(),"Scalar projection ciphertext metadata or alias differs");
 const auto& primes=context.get_context_data(input.chain_index()).parms().coeff_modulus();TORCH_CHECK(primes.size()==size_t(scalar.levels),"Scalar projection context level differs");
 for(int i=0;i<scalar.levels;++i)TORCH_CHECK(scalar.value[i]<primes[i].value()&&scalar.shoup[i]==uint64_t((__uint128_t(scalar.value[i])<<64)/primes[i].value()),"Invalid scalar RNS/Shoup residue");
 scalar_multiply_add<<<4096,256,0,phantom::util::global_variables::default_stream->get_stream()>>>(input.data(),output.data(),context.gpu_rns_tables().modulus(),scalar,65536,add);
 TORCH_CHECK(cudaGetLastError()==cudaSuccess,"Scalar projection kernel launch failed");
}
}
