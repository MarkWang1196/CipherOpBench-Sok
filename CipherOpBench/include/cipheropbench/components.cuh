#pragma once
#include "cipheropbench/operators.cuh"
namespace cipheropbench {
class ComponentOperators final : public TransformerOperators {
 std::string method_;
 [[noreturn]] void unsupported()const;
public:
 ComponentOperators(Runtime&,std::string);
 CipherList linear(CipherList,const LinearWeights&) override;
 QKV project_qkv(CipherList,const LayerWeights&) override;
 Cipher qk(Cipher,Cipher) override;
 Cipher av(Cipher,Cipher) override;
 Cipher softmax(Cipher,const k::FlatVec&) override;
 Cipher softmax_v(Cipher,Cipher,const k::FlatVec&) override;
 CipherList attention(CipherList,const LayerWeights&,torch::Tensor) override;
 CipherList gelu(CipherList) override;
 CipherList layer_norm(CipherList,torch::Tensor,torch::Tensor) override;
 CipherList ffn(CipherList,const LayerWeights&) override;
 CipherList block(CipherList,const LayerWeights&,torch::Tensor) override;
 CipherList encoder(CipherList,const std::vector<LayerWeights>&,torch::Tensor) override;
};
inline bool is_component_method(const std::string& s){return s=="private_llm"||s=="metakernel";}
bool component_scope_supported(const std::string&,const std::string&);
void register_component_methods();
}
