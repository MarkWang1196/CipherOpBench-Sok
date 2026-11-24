#pragma once
#include "cipheropbench/operators.cuh"
namespace cipheropbench {
struct ResearchPolynomial { double low,high; std::vector<double> coefficients; };
struct StipBatchProfile { double norm1_divisor,carry_divisor,norm2_divisor; };
struct AtlasLayer {
 int squares=0,newton[2]{},goldschmidt[2]{}; double maximum=0;
 ResearchPolynomial exponential,gelu; std::vector<ResearchPolynomial> inverse_sqrt;
 std::array<std::vector<double>,2> norm_scale;
};
class ResearchOperators final : public TransformerOperators {
 std::string method_; int layer_=0,norm_=1,pair_=0; std::vector<AtlasLayer> atlas_;
 Cipher carried_scale_; bool carrying_=false;
 bool published_domains_=false;
 Cipher distance(Cipher,Cipher);
 CipherList affine(CipherList,torch::Tensor,torch::Tensor);
public:
 ResearchOperators(Runtime&,std::string,const std::string& atlas_profile="",bool published_domains=false);
 CipherList linear(CipherList,const LinearWeights&) override;
 QKV project_qkv(CipherList,const LayerWeights&) override;
 Cipher qk(Cipher,Cipher) override;
 Cipher gaussian_scores(Cipher,Cipher);
 // STIP AMCP group: six heads, 32 samples per head, token stride 256.
 CipherList batch_stip_l2(CipherList query,CipherList key);
 CipherList batch_stip_gk(CipherList diagonals,torch::Tensor valid);
 CipherList batch_stip_av(CipherList diagonals,CipherList value);
 Cipher batch_stip_relu(Cipher input);
 std::array<CipherList,6> batch_stip_prepare(torch::Tensor input,torch::Tensor valid);
 std::array<CipherList,6> batch_stip_regroup(CipherList& input);
 CipherList batch_stip_encoder(CipherList input,const std::array<CipherList,6>& first_projection,
                              const std::vector<LayerWeights>& weights,torch::Tensor valid,
                              const std::vector<StipBatchProfile>& profiles);
 Cipher av(Cipher,Cipher) override;
 Cipher softmax(Cipher,const k::FlatVec&) override;
 CipherList gelu(CipherList) override;
 CipherList layer_norm(CipherList,torch::Tensor,torch::Tensor) override;
 CipherList attention(CipherList,const LayerWeights&,torch::Tensor) override;
 CipherList ffn(CipherList,const LayerWeights&) override;
 CipherList block(CipherList,const LayerWeights&,torch::Tensor) override;
 CipherList encoder(CipherList,const std::vector<LayerWeights>&,torch::Tensor) override;
};
bool is_research_method(const std::string&);
void register_research_methods(const std::string& atlas_profile="",bool published_domains=false);
}
