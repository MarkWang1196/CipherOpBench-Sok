#include "cipheropbench/operators.cuh"
#include <fstream>

namespace cipheropbench {
void Shape::validate_nexus() const {
  TORCH_CHECK(tokens==128 && hidden==768 && heads==12 && intermediate==3072,
              "NEXUS supports only tokens=128, hidden=768, heads=12, intermediate=3072");
}
static LinearWeights random_linear(int in, int out) {
  return {torch::randn({in,out},torch::kDouble)*0.02, torch::zeros({out},torch::kDouble)};
}
Fixture synthetic_fixture(int layers, int seed) {
  TORCH_CHECK(layers>=1 && layers<=12, "layers must be 1..12");
  torch::manual_seed(seed);
  Fixture f;
  f.evidence="synthetic";
  // Unit-variance inputs exercise the LN initialization domain.
  f.hidden=torch::randn({128,768},torch::kDouble);
  f.mask=torch::ones({128,128},torch::kBool);
  for(int i=0;i<layers;++i) {
    LayerWeights w{random_linear(768,768),random_linear(768,768),random_linear(768,768),
                   random_linear(768,768),random_linear(768,3072),random_linear(3072,768),
                   torch::ones({768},torch::kDouble),torch::zeros({768},torch::kDouble),
                   torch::ones({768},torch::kDouble),torch::zeros({768},torch::kDouble)};
    f.layers.push_back(std::move(w));
  }
  return f;
}
static torch::Tensor read_tensor(const std::filesystem::path& path, std::vector<int64_t> shape) {
  auto tensor=torch::empty(shape,torch::kDouble);
  auto bytes=tensor.numel()*sizeof(double);
  TORCH_CHECK(std::filesystem::exists(path) && std::filesystem::file_size(path)==bytes,
              "missing tensor or wrong shape: ",path.string());
  std::ifstream file(path,std::ios::binary);
  file.read(reinterpret_cast<char*>(tensor.data_ptr<double>()),bytes);
  TORCH_CHECK(file.good(),"failed to read ",path.string());
  TORCH_CHECK(torch::isfinite(tensor).all().item<bool>(),"non-finite fixture: ",path.string());
  return tensor;
}
Fixture load_fixture(const std::filesystem::path& directory, int layers) {
  TORCH_CHECK(layers>=0 && layers<=12,"fixture layers must be 0..12");
  Fixture f;
  f.evidence="model_trace";
  f.hidden=read_tensor(directory/"hidden.bin",{128,768});
  f.mask=read_tensor(directory/"mask.bin",{128,128}).to(torch::kBool);
  for(int i=0;i<layers;++i) {
    auto base=directory/("layer"+std::to_string(i));
    auto linear=[&](std::string name,int in,int out) {
      return LinearWeights{read_tensor(base/(name+".weight.bin"),{in,out}),read_tensor(base/(name+".bias.bin"),{out})};
    };
    f.layers.push_back({linear("q",768,768),linear("k",768,768),linear("v",768,768),linear("out",768,768),
                        linear("up",768,3072),linear("down",3072,768),
                        read_tensor(base/"ln1.weight.bin",{768}),read_tensor(base/"ln1.bias.bin",{768}),
                        read_tensor(base/"ln2.weight.bin",{768}),read_tensor(base/"ln2.bias.bin",{768})});
  }
  return f;
}
torch::Tensor plain_linear(torch::Tensor x,const LinearWeights& w) { return torch::mm(x,w.weight)+w.bias; }
torch::Tensor plain_attention(torch::Tensor x,torch::Tensor mask,const LayerWeights& w) {
  auto heads=[](torch::Tensor t){return t.view({128,12,64}).transpose(0,1);};
  auto q=heads(plain_linear(x,w.q)), key=heads(plain_linear(x,w.key)), v=heads(plain_linear(x,w.v));
  auto scores=torch::matmul(q,key.transpose(1,2))/8.;
  scores=scores.masked_fill(~mask.unsqueeze(0),-100.);
  auto values=torch::matmul(torch::softmax(scores,-1),v).transpose(0,1).contiguous().view({128,768});
  return plain_linear(values,w.out);
}
torch::Tensor plain_ffn(torch::Tensor x,const LayerWeights& w) {
  return plain_linear(torch::gelu(plain_linear(x,w.up)),w.down)+x;
}
torch::Tensor plain_block(torch::Tensor x,torch::Tensor mask,const LayerWeights& w) {
  auto attention=plain_attention(x,mask,w)+x;
  auto normalized=torch::layer_norm(attention,{768},w.ln1_weight,w.ln1_bias,1e-12);
  return torch::layer_norm(plain_ffn(normalized,w),{768},w.ln2_weight,w.ln2_bias,1e-12);
}
torch::Dict<torch::IValue,torch::IValue> native_state(const LayerWeights& w) {
  torch::Dict<torch::IValue,torch::IValue> d(c10::StringType::get(),c10::TensorType::get());
  auto put=[&](std::string prefix,const LinearWeights& p) {
    d.insert(prefix+".weight",p.weight.t().to(torch::kFloat).contiguous());
    d.insert(prefix+".bias",p.bias.to(torch::kFloat));
  };
  put("bert.encoder.layer.0.attention.self.query",w.q);
  put("bert.encoder.layer.0.attention.self.key",w.key);
  put("bert.encoder.layer.0.attention.self.value",w.v);
  put("bert.encoder.layer.0.attention.output.dense",w.out);
  put("bert.encoder.layer.0.intermediate.dense",w.up);
  put("bert.encoder.layer.0.output.dense",w.down);
  d.insert("bert.encoder.layer.0.attention.output.LayerNorm.weight",w.ln1_weight.to(torch::kFloat));
  d.insert("bert.encoder.layer.0.attention.output.LayerNorm.bias",w.ln1_bias.to(torch::kFloat));
  d.insert("bert.encoder.layer.0.output.LayerNorm.weight",w.ln2_weight.to(torch::kFloat));
  d.insert("bert.encoder.layer.0.output.LayerNorm.bias",w.ln2_bias.to(torch::kFloat));
  return d;
}
}
