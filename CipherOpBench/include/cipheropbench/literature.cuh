#pragma once
#include "cipheropbench/operators.cuh"
namespace cipheropbench {
struct EustonSoftmaxProfile {
  double shift, divisor;
  int squarings, inverse_iterations;
};
// Existing ciphertext types and public operator signatures are unchanged.
class LiteratureOperators final : public TransformerOperators {
  std::string method_;
  bool published_domains_=false;
  int layer_=0, norm_kind_=1;
public:
  LiteratureOperators(Runtime& runtime,std::string method,bool published_domains=false);
  CipherList linear(CipherList x,const LinearWeights& w) override;
  QKV project_qkv(CipherList x,const LayerWeights& w) override;
  Cipher qk(Cipher q,Cipher key) override;
  CipherList batch_qk(const CipherList& query,const CipherList& key);
  Cipher av(Cipher p,Cipher value) override;
  CipherList batch_av(const CipherList& diagonals,const CipherList& values);
  CipherList batch_softmax(CipherList scores,const torch::Tensor& valid,const EustonSoftmaxProfile& profile);
  Cipher batch_gelu(Cipher input);
  Cipher batch_gelu(Cipher input,double input_bound);
  CipherList batch_layer_norm(CipherList input,torch::Tensor weight,torch::Tensor bias,
                             const std::vector<double>& slot_mask,double divisor);
  Cipher softmax(Cipher x,const k::FlatVec& mask) override;
  CipherList gelu(CipherList x) override;
  CipherList layer_norm(CipherList x,torch::Tensor w,torch::Tensor b) override;
  CipherList block(CipherList x,const LayerWeights& w,torch::Tensor mask) override;
  CipherList encoder(CipherList x,const std::vector<LayerWeights>& w,torch::Tensor mask) override;
};
void register_literature_methods(bool published_domains=false);
}
