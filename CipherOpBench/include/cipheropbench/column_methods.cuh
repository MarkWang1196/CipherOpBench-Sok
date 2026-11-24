#pragma once
#include "cipheropbench/operators.cuh"
#include <utility>
namespace cipheropbench {
struct ColumnProfile {
 int inverse_iterations=2;
 double statmax=2.08,exp_low=-60.,exp_high=15.,inverse_low=.02,inverse_high=400.,gelu_low=-61.,gelu_high=136.,norm1_low=.768,norm1_high=1612.8,norm2_low=.5,norm2_high=2500.;
};
struct ArionSoftmaxReference {
 torch::Tensor exponentials,unnormalized_av,normalized_av;
};
// Published column/diagonal algorithms with canonical encrypted boundaries.
class ColumnOperators final : public TransformerOperators {
 std::string method_; int layer_=0, norm_=1;
 std::array<ColumnProfile,12> profiles_;
public:
 ColumnOperators(Runtime&,std::string,std::string profile="");
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
 // ARION's full column slots are indexed by (token, independent input).
 static constexpr int batch_capacity = 256;
 CipherList encrypt_batch(torch::Tensor hidden) const;
 CipherList encrypt_batch(torch::Tensor hidden,size_t chain) const;
 torch::Tensor decrypt_batch(const CipherList& hidden, int active_inputs) const;
 CipherList batch_linear(CipherList hidden, const LinearWeights& weights);
 CipherList batch_qk(CipherList query, CipherList key);
 // ARION consumes giant-shifted exponentials; MOAI consumes canonical probability diagonals.
 CipherList batch_av(CipherList attention, CipherList value);
 CipherList batch_softmax(CipherList scores, torch::Tensor valid);
 CipherList batch_layer_norm(CipherList hidden,torch::Tensor weight,torch::Tensor bias,
                            torch::Tensor valid,int kind);
 std::pair<double,double> batch_arion_gelu_domain() const;
 Cipher batch_arion_gelu(Cipher input,const std::vector<double>& slot_mask);
 static constexpr double moai_batch_gelu_bound=21.;
 static constexpr double arion_batch_score_bound=16., arion_batch_exp_low=-24., arion_batch_exp_high=6.;
 torch::Tensor moai_batch_gelu_reference(const torch::Tensor& input) const;
 Cipher batch_moai_gelu(Cipher input,const std::vector<double>& slot_mask);
 ArionSoftmaxReference arion_softmax_reference(const torch::Tensor& scores,const torch::Tensor& values,
                                              const torch::Tensor& valid) const;
 CipherList batch_arion_softmax(CipherList scores,CipherList unnormalized_av,
                               const torch::Tensor& valid,int head);
 CipherList batch_encoder(CipherList hidden, const std::vector<LayerWeights>& weights, torch::Tensor valid);

};
inline bool is_column_method(const std::string& s){return s=="arion"||s=="moai";}
void register_column_methods(const std::string& profile="");
}
