#pragma once
#include "cipheropbench/runtime.cuh"
namespace cipheropbench {
// Independent from BERT's fixed 128x768 ABI. All dimensions and positions are public.
struct DecoderRows { CipherList blocks; int rows=0,width=0,stride=0; };
class DecoderOperators {
 friend class EncryptedKVCache;
 Runtime& runtime_;
public:
 explicit DecoderOperators(Runtime& r):runtime_(r){}
 DecoderRows encrypt_rows(torch::Tensor x); // client boundary, top chain
 DecoderRows encrypt_rows(torch::Tensor x,double input_scale); // explicit common40 or ARION input45
 torch::Tensor decrypt_rows(const DecoderRows& x); // client verification only
 DecoderRows rope(const DecoderRows&,int start_position,int head_dim=128,double theta=500000.);
 DecoderRows rmsnorm(const DecoderRows&,torch::Tensor gamma,double epsilon=1e-5,int arion_kind=0);
 DecoderRows swiglu(const DecoderRows& gate,const DecoderRows& up);
 DecoderRows arion_exp(const DecoderRows&);
 DecoderRows arion_inverse(const DecoderRows&);
 DecoderRows qk(const DecoderRows& query,const DecoderRows& key,int query_start);
 DecoderRows softmax(const DecoderRows& scores,int query_tokens,int query_start,int cache_tokens);
 DecoderRows av(const DecoderRows& probabilities,const DecoderRows& value,int query_tokens);
 DecoderRows append(const DecoderRows& past,const DecoderRows& incoming);
};
class EncryptedKVCache {
 DecoderOperators& ops_; int capacity_,retained_levels_;
public:
 DecoderRows key,value; int length=0;
 // Public modulus ceiling; lower-budget inputs retain their own level budget.
 explicit EncryptedKVCache(DecoderOperators& ops,int capacity=8192,int retained_levels=31):ops_(ops),capacity_(capacity),retained_levels_(retained_levels){TORCH_CHECK(capacity>0&&retained_levels>=4&&retained_levels<=31,"Invalid KV capacity or retained level ceiling");}
 void append(const DecoderRows& k,const DecoderRows& v,int start_position);
};
}
