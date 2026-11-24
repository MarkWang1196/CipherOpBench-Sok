#pragma once
#include "cipheropbench/operators.cuh"
namespace cipheropbench {
class ExtendedOperators final : public TransformerOperators {
  std::string method_, statistics_path_;
  std::vector<std::array<std::array<std::vector<double>,2>,2>> statistics_;
public:
  ExtendedOperators(Runtime&,std::string method,std::string statistics="");
  CipherList linear(CipherList,const LinearWeights&) override;
  QKV project_qkv(CipherList,const LayerWeights&) override;
  Cipher qk(Cipher,Cipher) override;
  Cipher av(Cipher,Cipher) override;
  Cipher softmax(Cipher,const k::FlatVec&) override;
  CipherList gelu(CipherList) override;
  CipherList layer_norm(CipherList,torch::Tensor,torch::Tensor) override;
  CipherList ffn(CipherList,const LayerWeights&) override;
  CipherList block(CipherList,const LayerWeights&,torch::Tensor) override;
  CipherList encoder(CipherList,const std::vector<LayerWeights>&,torch::Tensor) override;
};
bool extended_scope_supported(const std::string&,const std::string&);
void register_extended_methods(const std::string& statistics="");
}
