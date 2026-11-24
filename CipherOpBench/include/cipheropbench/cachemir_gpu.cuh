#pragma once
#include "ckks_evaluator.cuh"
#include "Bootstrapper.cuh"
#include "cipheropbench/unified_phantom_parameters.cuh"
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cipheropbench::cachemir {
using Cipher = PhantomCiphertext;
using Values = std::vector<double>;
using Weight = std::function<double(int,int)>;
constexpr int slots=32768, hidden=4096, intermediate=14336;
constexpr double scale=unified_phantom::scale;

// Public evaluation state is built here, rather than accepting a client Runtime.
// Its evaluator never receives a secret key or a decrypt/re-encrypt callback.
class GPU {
  std::shared_ptr<nexus::CKKSEvaluator> public_;
  std::shared_ptr<::Bootstrapper> bootstrapper_;
  double maximum_product_scale_gain_error_=0;
public:
  GPU(std::shared_ptr<PhantomContext> context,
      std::shared_ptr<PhantomPublicKey> pk,
      std::shared_ptr<PhantomCKKSEncoder> encoder,
      std::shared_ptr<PhantomRelinKey> relin,
      std::shared_ptr<PhantomGaloisKey> rotations, bool bootstrap);
  static void synchronize();
  static void need(const Cipher& x,int moduli=0);
  Cipher drop(Cipher x,int moduli);
  Cipher rotate(Cipher x,int shift);
  Cipher add(Cipher a,Cipher b);
  Cipher add_constant(Cipher x,double value);
  Cipher add_plain(Cipher x,const Values& values);
  Cipher negate(Cipher x);
  Cipher multiply_plain(Cipher x,const Values& values);
  Cipher multiply_constant(Cipher x,double value);
  Cipher product(Cipher a,Cipher b);
  Cipher multiply_integer(Cipher x,int value);
  Cipher clean(Cipher x,int rows,double factor=1.);
  Cipher rms(Cipher x,const Values& gamma,double output_factor=1.);
  Cipher linear(Cipher x,const Weight& weight);
  std::vector<Cipher> project(const std::vector<Cipher>& inputs,int in_width,
                              int out_width,const Weight& weight);
  Cipher rope(Cipher x,int position);
  Cipher silu(Cipher x,const Values& chebyshev_coefficients);
  Cipher refresh(Cipher x,double public_amplitude=unified_phantom::refresh_amplitude,int output_moduli=unified_phantom::working_moduli,int leading_rows=0);
  Cipher refresh_scaled(Cipher x,double public_amplitude=unified_phantom::refresh_amplitude,int output_moduli=unified_phantom::working_moduli,int leading_rows=0);
  double maximum_product_scale_gain_error()const{return maximum_product_scale_gain_error_;}
};
struct Page {
  std::map<int,std::shared_ptr<const Cipher>> k;
  std::array<std::shared_ptr<const Cipher>,128> v;
};
struct Cache {
  int next=0;
  std::map<int,std::shared_ptr<const Page>> pages;
};
struct LayerWeights {Weight q,k,v,o,gate,up,down;Values gamma_attention,gamma_ffn,silu_coefficients;};
struct LayerResult {
  Cipher output;
  std::map<std::string,Cipher> trace;
  std::map<std::string,double> stage_ms;
  int bootstrap_calls=0;
};
class Layer {
  GPU& gpu_;
  LayerWeights weights_;
  Cache cache_;
  using Scores=std::array<std::vector<Cipher>,4>;
  void append(Cache& cache,int position,Cipher k,Cipher v);
  Scores qk(const Cache& cache,Cipher q);
  Scores normalize(const Cache& cache,const Scores& scores,int& bootstrap_calls);
  Cipher av(const Cache& cache,const Scores& probabilities);
public:
  Layer(GPU& gpu,LayerWeights weights);
  LayerResult apply(Cipher input,int position);
  // Diagnostics retain intermediate ciphertexts and synchronize stages; disable for online timing.
  LayerResult apply(Cipher input,int position,bool diagnostics);
  const Cache& cache()const{return cache_;}
};
}
