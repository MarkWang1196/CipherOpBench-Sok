#pragma once
#include "cipheropbench/runtime.cuh"

namespace cipheropbench::nexus_batch {
// PUMA Eq. (3) supplies the two middle polynomials in NEXUS Eq. (3).
inline const std::vector<double> gelu_lower{-0.5054031199708174,-0.42226581151983866,
                                            -0.11807612951181953,-0.011034134030615728};
inline const std::vector<double> gelu_upper{0.008526321541038084,.5,0.3603292692789629,
                                            0.,-0.037688200365904236,0.,0.0018067462606141187};
inline constexpr double gelu_sign_divisor=32.;
inline constexpr int softmax_exponential_gain_bits(int inverse_iterations,double divisor) {
  const bool amplify=inverse_iterations>=15||(inverse_iterations>4&&divisor<=.125);
  return amplify?inverse_iterations-4:0;
}
inline constexpr int softmax_denominator_gain_bits(int exponential_gain_bits) {
  return exponential_gain_bits>0?6:0;
}
Cipher gelu(Runtime& runtime,Cipher input);
Cipher softmax(Runtime& runtime,Cipher scores,std::vector<double> mask,
               double shift,double divisor,int inverse_iterations);

inline void add_client_share(PhantomCiphertext& encrypted, PhantomSecretKey& client,
                             const PhantomContext& context, ::nexus::Encoder& encoder) {
  TORCH_CHECK(encrypted.size()==2&&encrypted.is_ntt_form(),"Expected a two-polynomial CKKS ciphertext");
  const auto& stream=*phantom::util::global_variables::default_stream;
  const auto bytes=encrypted.poly_modulus_degree()*encrypted.coeff_modulus_size()*sizeof(uint64_t);
  auto share=encrypted;
  auto error=cudaMemsetAsync(share.data(0),0,bytes,stream.get_stream());
  TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
  PhantomPlaintext client_product;
  client.decrypt(context,share,client_product);
  // Reuse Phantom's symmetric sampler for fresh coefficient noise. No slots are decoded.
  PhantomPlaintext zero;
  encoder.encode(0.,encrypted.chain_index(),encrypted.scale(),zero);
  PhantomCiphertext fresh;
  client.encrypt_symmetric(context,zero,fresh);
  PhantomPlaintext noise;
  client.decrypt(context,fresh,noise);
  // Phantom decrypts c0 + c1*s. Subtract c1*s_client before the server removes its share.
  ::sub_plain_inplace(context,encrypted,client_product);
  ::add_plain_inplace(context,encrypted,noise);
}

inline void remove_server_share(PhantomCiphertext& encrypted, PhantomSecretKey& server,
                                const PhantomContext& context) {
  PhantomPlaintext remaining;
  server.decrypt(context,encrypted,remaining);
  const auto bytes=encrypted.poly_modulus_degree()*encrypted.coeff_modulus_size()*sizeof(uint64_t);
  const auto& stream=*phantom::util::global_variables::default_stream;
  auto error=cudaMemcpyAsync(encrypted.data(0),remaining.data(),bytes,cudaMemcpyDeviceToDevice,stream.get_stream());
  TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
}
}
