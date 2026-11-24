#pragma once
#include "cipheropbench/runtime.cuh"
#include "cipheropbench/kernels/matrix_mul.cuh"
#include "cipheropbench/kernels/softmax.cuh"
#include "cipheropbench/kernels/gelu.cuh"
#include "cipheropbench/kernels/layer_norm.cuh"
#include <array>
#include <filesystem>

namespace cipheropbench {
// The NEXUS arithmetic schedule has a fixed physical contract. New methods can
// implement the same interface with different packing and shape contracts.
struct Shape {
  int tokens = 128, hidden = 768, heads = 12, intermediate = 3072;
  void validate_nexus() const;
};
struct LinearWeights { torch::Tensor weight, bias; }; // [in, out], [out]
struct LayerWeights {
  LinearWeights q, key, v, out, up, down;
  torch::Tensor ln1_weight, ln1_bias, ln2_weight, ln2_bias;
};
struct Fixture {
  Shape shape;
  torch::Tensor hidden, mask;
  std::vector<LayerWeights> layers;
  std::string evidence;
};
Fixture synthetic_fixture(int layers, int seed);
Fixture load_fixture(const std::filesystem::path& directory, int layers);
torch::Tensor plain_linear(torch::Tensor x, const LinearWeights& weights);
torch::Tensor plain_attention(torch::Tensor x, torch::Tensor mask, const LayerWeights& weights);
torch::Tensor plain_ffn(torch::Tensor x, const LayerWeights& weights);
torch::Tensor plain_block(torch::Tensor x, torch::Tensor mask, const LayerWeights& weights);
torch::Dict<torch::IValue, torch::IValue> native_state(const LayerWeights& weights);

struct QKV {
  std::array<Cipher, 6> q, key, v;
};
class TransformerOperators {
protected:
  Runtime& runtime_;
  int calibration_layer_=0, calibration_norm_=1;
  k::MMEvaluator matmul_;
  k::SoftmaxEvaluator softmax_;
  k::GELUEvaluator gelu_;
  k::LNEvaluator norm_;
public:
  explicit TransformerOperators(Runtime& runtime);
  virtual ~TransformerOperators() = default;
  // Public composable levels: no decrypt/re-encrypt inside these methods.
  virtual QKV project_qkv(CipherList x, const LayerWeights& weights);
  virtual Cipher qk(Cipher q, Cipher key);
  virtual Cipher softmax(Cipher scores, const k::FlatVec& diagonal_mask);
  virtual Cipher av(Cipher probabilities, Cipher value);
  virtual Cipher softmax_v(Cipher scores, Cipher value, const k::FlatVec& mask);
  virtual CipherList attention(CipherList x, const LayerWeights& weights, torch::Tensor mask);
  virtual CipherList linear(CipherList x, const LinearWeights& weights);
  virtual CipherList gelu(CipherList x);
  virtual CipherList layer_norm(CipherList x, torch::Tensor weight, torch::Tensor bias);
  virtual CipherList ffn(CipherList x, const LayerWeights& weights);
  virtual CipherList block(CipherList x, const LayerWeights& weights, torch::Tensor mask);
  virtual CipherList encoder(CipherList x, const std::vector<LayerWeights>& weights, torch::Tensor mask);
  virtual torch::Tensor unpack_qk(const Cipher& x) const;
  virtual torch::Tensor unpack_av(const Cipher& x) const;
  virtual torch::Tensor unpack_qkv(const QKV& x) const;
  static torch::Tensor diagonalize(torch::Tensor matrix);
  static k::FlatVec pack_score_mask(torch::Tensor mask);
};
}
