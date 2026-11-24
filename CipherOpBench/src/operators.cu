#include "cipheropbench/operators.cuh"
#include "calibrated_operators.cuh"

namespace cipheropbench {
static constexpr std::array<std::array<int,2>,6> head_pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
TransformerOperators::TransformerOperators(Runtime& r):runtime_(r),matmul_(r.ckks),softmax_(r.ckks,r.bootstrapper),gelu_(r.ckks,r.bootstrapper),norm_(r.ckks,r.bootstrapper) {}

QKV TransformerOperators::project_qkv(CipherList x,const LayerWeights& w) {
  for(auto& ct:x) runtime_.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(6));
  QKV out;
  for(int i=0;i<6;++i) {
    auto [a,b]=head_pairs[i];
    auto slice=[&](const torch::Tensor& t,int h){return t.slice(-1,h*64,(h+1)*64);};
    auto project_pair=[&](const LinearWeights& p,Cipher& destination) {
      auto weights=k::row_pack_768x64x2(slice(p.weight,a),slice(p.weight,b));
      matmul_.matrix_mul_ct128x768_pt768x64x2(x,weights,destination);
      auto bias=k::row_pack_64x1x2(slice(p.bias,a),slice(p.bias,b));
      auto pt=k::CKKSEncode(bias,runtime_.ckks,&destination);
      runtime_.ckks->evaluator.add_plain_inplace(destination,pt);
    };
    runtime_.record("q_projection",[&]{project_pair(w.q,out.q[i]);},x[0].chain_index(),&out.q[i]);
    runtime_.record("k_projection",[&]{project_pair(w.key,out.key[i]);},x[0].chain_index(),&out.key[i]);
    runtime_.record("v_projection",[&]{
      auto weights=k::row_pack_768x128(torch::cat({slice(w.v.weight,a),slice(w.v.weight,b)},1));
      matmul_.matrix_mul_ct128x768_pt768x128(x,weights,out.v[i]);
      auto pt=k::CKKSEncode(k::row_pack_128x1(torch::cat({slice(w.v.bias,a),slice(w.v.bias,b)},0)),runtime_.ckks,&out.v[i]);
      runtime_.ckks->evaluator.add_plain_inplace(out.v[i],pt);
    },x[0].chain_index(),&out.v[i]);
  }
  return out;
}
Cipher TransformerOperators::qk(Cipher q,Cipher key) {
  Cipher out;
  runtime_.record("qk",[&]{
    matmul_.matrix_mul_ct128x64_ct128x64_transpose(q,key,out);
    auto ratio=k::CKKSEncode(std::vector<double>(k::slot_count,0.125),runtime_.ckks,&out,true);
    runtime_.ckks->evaluator.multiply_plain_inplace(out,ratio);
    runtime_.ckks->evaluator.rescale_to_next_inplace(out);
  },q.chain_index(),&out);
  return out;
}
Cipher TransformerOperators::softmax(Cipher scores,const k::FlatVec& mask) {
  if(runtime_.calibration.enabled)return detail::calibrated_softmax(runtime_,scores,mask,calibration_layer_);
  Cipher out;
  runtime_.record("softmax_with_refresh",[&]{
    runtime_.refresh(scores);
    softmax_.softmax_128x128(scores,out,mask);
  },scores.chain_index(),&out);
  return out;
}
Cipher TransformerOperators::av(Cipher probabilities,Cipher value) {
  Cipher out;
  runtime_.record("av",[&]{matmul_.matrix_mul_ct128x128_ct128x128(probabilities,value,out);},probabilities.chain_index(),&out);
  return out;
}
Cipher TransformerOperators::softmax_v(Cipher scores,Cipher value,const k::FlatVec& mask) { return av(softmax(scores,mask),value); }
torch::Tensor TransformerOperators::diagonalize(torch::Tensor matrix) {
  auto out=matrix.clone();
  for(int i=0;i<128;++i) out[i]=torch::roll(matrix[i],-i,0);
  return out;
}
k::FlatVec TransformerOperators::pack_score_mask(torch::Tensor mask) {
  auto additive=torch::zeros({128,128},torch::kDouble).masked_fill(~mask,-100.);
  auto diagonal=diagonalize(additive);
  return k::flatten_pack(diagonal,diagonal);
}
CipherList TransformerOperators::attention(CipherList x,const LayerWeights& w,torch::Tensor mask) {
  auto projected=project_qkv(x,w);
  auto packed_mask=pack_score_mask(mask);
  std::array<Cipher,6> output;
  for(int i=0;i<6;++i) output[i]=softmax_v(qk(projected.q[i],projected.key[i]),projected.v[i],packed_mask);
  CipherList merged(3);
  runtime_.record("merge_heads",[&]{
    for(int i=0;i<3;++i) {
      runtime_.ckks->evaluator.rotate_vector_inplace(output[2*i+1],k::slot_count/2,*runtime_.ckks->galois_keys);
      runtime_.ckks->evaluator.add(output[2*i],output[2*i+1],merged[i]);
    }
  });
  return linear(merged,w.out);
}
CipherList TransformerOperators::linear(CipherList x,const LinearWeights& w) {
  TORCH_CHECK(w.weight.sizes()==torch::IntArrayRef({768,768}),"linear physical block requires [768,768]");
  auto packed=k::row_pack_768x768(w.weight);
  auto biases=k::row_pack_768x1(w.bias);
  CipherList out;
  runtime_.record("linear_768",[&]{
    matmul_.matrix_mul_ct128x768_pt768x768(x,packed,out);
    for(int i=0;i<3;++i) {auto pt=k::CKKSEncode(biases[i],runtime_.ckks,&out[i]);runtime_.ckks->evaluator.add_plain_inplace(out[i],pt);}
  },x[0].chain_index());
  return out;
}
CipherList TransformerOperators::gelu(CipherList x) {
  if(runtime_.calibration.enabled)return detail::calibrated_gelu(runtime_,std::move(x),calibration_layer_);
  CipherList out(x.size());
  for(size_t i=0;i<x.size();++i) runtime_.record("gelu",[&]{gelu_.gelu(x[i],out[i]);},x[i].chain_index(),&out[i]);
  return out;
}
CipherList TransformerOperators::layer_norm(CipherList x,torch::Tensor weight,torch::Tensor bias) {
  if(runtime_.calibration.enabled)return detail::calibrated_norm(runtime_,std::move(x),weight,bias,calibration_layer_,calibration_norm_);
  auto [packed_w,packed_b]=k::row_pack_layer_norm(weight,bias);
  CipherList out;
  runtime_.record("layer_norm",[&]{norm_.layer_norm_128x768(x,out,packed_w,packed_b);},x[0].chain_index());
  return out;
}
CipherList TransformerOperators::ffn(CipherList x,const LayerWeights& w) {
  CipherList residual=x,out;
  for(int block=0;block<4;++block) {
    LinearWeights up{w.up.weight.slice(1,block*768,(block+1)*768),w.up.bias.slice(0,block*768,(block+1)*768)};
    auto activated=gelu(linear(x,up));
    LinearWeights down{w.down.weight.slice(0,block*768,(block+1)*768),torch::zeros({768},torch::kDouble)};
    auto part=linear(activated,down);
    if(block==0) out=std::move(part);
    else for(int i=0;i<3;++i) runtime_.ckks->evaluator.add_inplace(out[i],part[i]);
  }
  auto bias=k::row_pack_768x1(w.down.bias);
  runtime_.record("ffn_bias_residual",[&]{
    for(int i=0;i<3;++i) {
      auto pt=k::CKKSEncode(bias[i],runtime_.ckks,&out[i]);
      runtime_.ckks->evaluator.add_plain_inplace(out[i],pt);
      k::mod_switch_to_same(residual[i],out[i],runtime_.ckks);
      runtime_.ckks->evaluator.add_inplace(out[i],residual[i]);
    }
  });
  return out;
}
CipherList TransformerOperators::block(CipherList x,const LayerWeights& w,torch::Tensor mask) {
  detail::calibrated_mask(runtime_,x,mask);
  auto result=attention(x,w,mask);
  runtime_.record("attention_residual",[&]{
    for(int i=0;i<3;++i) {k::mod_switch_to_same(x[i],result[i],runtime_.ckks);runtime_.ckks->evaluator.add_inplace(result[i],x[i]);}
  });
  runtime_.refresh(result);
  for(auto& ct:result) runtime_.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(6));
  calibration_norm_=1;result=layer_norm(result,w.ln1_weight,w.ln1_bias);
  runtime_.refresh(result);
  for(auto& ct:result) runtime_.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(14));
  result=ffn(result,w);
  calibration_norm_=2;result=layer_norm(result,w.ln2_weight,w.ln2_bias);
  runtime_.refresh(result);
  calibration_norm_=1;return result;
}
CipherList TransformerOperators::encoder(CipherList x,const std::vector<LayerWeights>& weights,torch::Tensor mask) {
  for(size_t layer=0;layer<weights.size();++layer) {
    calibration_layer_=layer;x=block(x,weights[layer],mask);
    if(runtime_.layer_completed) runtime_.layer_completed(layer+1,weights.size());
  }
  calibration_layer_=0;calibration_norm_=1;return x;
}
torch::Tensor TransformerOperators::unpack_qk(const Cipher& x) const {
  auto t=runtime_.decrypt_slots(x);
  for(int h=0;h<2;++h) for(int row=0;row<128;++row) t[h][row]=torch::roll(t[h][row],row,0);
  return t;
}
torch::Tensor TransformerOperators::unpack_av(const Cipher& x) const {
  auto t=runtime_.decrypt_slots(x);
  return torch::stack({t[0].slice(1,0,64),t[1].slice(1,64,128)});
}
torch::Tensor TransformerOperators::unpack_qkv(const QKV& x) const {
  std::vector<torch::Tensor> groups;
  for(int kind=0;kind<3;++kind) {
    std::vector<torch::Tensor> heads(12);
    for(int pair=0;pair<6;++pair) {
      const auto& ct=kind==0?x.q[pair]:kind==1?x.key[pair]:x.v[pair];
      auto t=runtime_.decrypt_slots(ct);
      auto [a,b]=head_pairs[pair];
      heads[a]=t[0].slice(1,0,64);
      heads[b]=kind==2?t[0].slice(1,64,128):t[1].slice(1,0,64);
    }
    groups.push_back(torch::cat(heads,1));
  }
  return torch::stack(groups);
}
}
