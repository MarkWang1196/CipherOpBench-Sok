#pragma once
#include "cipheropbench/operators.cuh"
namespace cipheropbench {
class PublishedOperators final : public TransformerOperators {
 std::string method_;
 [[noreturn]] void unsupported()const;
public:
 PublishedOperators(Runtime&,std::string);
 CipherList linear(CipherList,const LinearWeights&) override;
 QKV project_qkv(CipherList,const LayerWeights&) override;
 Cipher qk(Cipher,Cipher) override;
 Cipher av(Cipher,Cipher) override;
 Cipher qk_av(Cipher,Cipher,Cipher);
 Cipher softmax(Cipher,const k::FlatVec&) override;
 Cipher softmax_v(Cipher,Cipher,const k::FlatVec&) override;
 CipherList attention(CipherList,const LayerWeights&,torch::Tensor) override;
 CipherList gelu(CipherList) override;
 CipherList layer_norm(CipherList,torch::Tensor,torch::Tensor) override;
 CipherList ffn(CipherList,const LayerWeights&) override;
 CipherList block(CipherList,const LayerWeights&,torch::Tensor) override;
 CipherList encoder(CipherList,const std::vector<LayerWeights>&,torch::Tensor) override;
 static constexpr int block_batch_size=4;
 static constexpr int block_softmax_working_scale_bits=46;
 static constexpr double block_softmax_shift=3.,block_softmax_divisor=65536.;
 CipherList encrypt_block_batch(const torch::Tensor& input,int levels=8) const;
 torch::Tensor decrypt_block_batch(const CipherList& input) const;
 CipherList batch_block_linear(CipherList& input,const LinearWeights& weights);
 CipherList batch_block_qk(Cipher query,Cipher key);
 Cipher batch_block_av(const CipherList& probabilities,Cipher value);
 CipherList batch_block_softmax(CipherList input,const k::FlatVecArray& shift);
 CipherList batch_block_gelu(CipherList input,double linear,double bias);
 CipherList batch_block_norm(CipherList input,const torch::Tensor& bias);
};
inline bool is_published_method(const std::string& s){return s=="blockformer"||s=="encformer"||s=="cgf_softmax";}
bool published_scope_supported(const std::string&,const std::string&);
void register_published_methods();
}
