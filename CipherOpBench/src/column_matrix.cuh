#pragma once
#include "research_math.cuh"
#include "arion_scalar_kernel.cuh"
namespace cipheropbench::detail {
constexpr int column_baby=12,column_giants=11;
// One logical sample, repeated in 256 blocks of 128 slots. This is the
// single-sample specialization of interleaved batching; copies are never samples.
inline void column_ready(Math& m,CipherList& x,int depth=5,int keep=8){
 for(auto& c:x){m.need(c,depth);if(c.coeff_modulus_size()>size_t(keep))m.r.ckks->evaluator.mod_switch_to_inplace(c,k::chain_idx(keep));}
 size_t chain=0;for(auto& c:x)chain=std::max(chain,c.chain_index());for(auto& c:x){m.r.ckks->evaluator.mod_switch_to_inplace(c,chain);c.set_scale(k::scale);}
}
inline CipherList hidden_columns(Math& m,CipherList x){
 TORCH_CHECK(x.size()==3,"Expected three canonical hidden ciphertexts");
 for(auto& c:x){m.need(c,6);c=shallow_tile_transpose(m,c);}CipherList out;out.reserve(768);
 for(int f=0;f<768;++f)out.push_back(column(m,x,f));column_ready(m,out);return out;
}
inline CipherList pair_columns(Math& m,Cipher x,int head,int width,bool value=false,int keep=8){
 TORCH_CHECK((head==0||head==1)&&(width==64||width==128),"Invalid head column dimensions");
 m.need(x,6);x=shallow_tile_transpose(m,x);CipherList out;out.reserve(width);
 for(int f=0;f<width;++f)out.push_back(column(m,{x},head*128+(value?head*64:0)+f));column_ready(m,out,5,keep);return out;
}
inline Cipher column_finish_dot(Math& m,Cipher sum,double factor=1.){
 m.r.ckks->evaluator.relinearize_inplace(sum,*m.r.ckks->relin_keys);m.r.ckks->evaluator.rescale_to_next_inplace(sum);
 // Preserve the actual S^2/q scale, then normalize using the next prime.
 return m.pm(std::move(sum),Vec(slots,factor));
}
inline Cipher column_dot(Math& m,const CipherList& a,const CipherList& b,double factor=1.){
 TORCH_CHECK(!a.empty()&&a.size()==b.size(),"Column dot product dimensions differ");Cipher sum;
 for(size_t j=0;j<a.size();++j){auto x=a[j],y=b[j];m.align(x,y);Cipher term;m.r.ckks->evaluator.multiply(x,y,term);if(j==0)sum=std::move(term);else m.r.ckks->evaluator.add_inplace(sum,term);}
 return column_finish_dot(m,std::move(sum),factor);
}
inline CipherList column_cpmm(Math& m,CipherList& input,const LinearWeights& w,int required_depth=5){
 TORCH_CHECK(w.weight.dim()==2&&w.weight.size(0)==int64_t(input.size())&&w.bias.numel()==w.weight.size(1),"Column CPMM dimensions differ");
 column_ready(m,input,required_depth);auto weights=w.weight.contiguous();auto a=weights.accessor<double,2>();CipherList result;result.reserve(weights.size(1));
 double prime=m.r.ckks->context->get_context_data(input[0].chain_index()).parms().coeff_modulus().back().value();
 const char* scalar=std::getenv("CIPHEROPBENCH_COLUMN_SCALAR_RNS");
 if(m.column_token_stride>1&&scalar&&std::string(scalar)=="1"){
  const auto& context=*m.r.ckks->context;
  const auto& primes=context.get_context_data(input[0].chain_index()).parms().coeff_modulus();
  std::vector<uint64_t> moduli;
  for(const auto& p:primes)moduli.push_back(p.value());
  const double plain_scale=prime*k::scale/input[0].scale();
  for(int j=0;j<weights.size(1);++j){
   auto sum=input[0];bool initialized=false;
   for(int i=0;i<weights.size(0);++i){
    if(a[i][j]==0.)continue;
    arion_scalar_accumulate(context,input[i],sum,arion_scalar_rns(a[i][j],plain_scale,moduli),initialized);
    initialized=true;
   }
   if(initialized){
    sum.set_scale(input[0].scale()*plain_scale);
    m.r.ckks->evaluator.rescale_to_next_inplace(sum);
    sum.set_scale(k::scale);
   }else sum=m.zero(input[0]);
   result.push_back(m.addc(std::move(sum),w.bias[j].item<double>()));
  }
  return result;
 }
 for(int j=0;j<weights.size(1);++j){Cipher sum;bool first=true;
  for(int i=0;i<weights.size(0);++i){if(a[i][j]==0.)continue;PhantomPlaintext pt;
   m.r.ckks->encoder.encode(Vec(slots,a[i][j]),input[i].chain_index(),prime*k::scale/input[i].scale(),pt);
   Cipher term;m.r.ckks->evaluator.multiply_plain(input[i],pt,term);if(first)sum=std::move(term);else m.r.ckks->evaluator.add_inplace(sum,term);first=false;
  }
  if(first)sum=m.zero(input[0]);else{m.r.ckks->evaluator.rescale_to_next_inplace(sum);sum.set_scale(k::scale);}
  result.push_back(m.addc(std::move(sum),w.bias[j].item<double>()));
 }return result;
}
inline CipherList column_hidden(Math& m,CipherList cols){
 // Mask first, then refresh the three packed ciphertexts inside the transpose.
 TORCH_CHECK(cols.size()==768,"Expected 768 output columns");column_ready(m,cols,1);CipherList result;
 for(int group=0;group<3;++group){Cipher sum;
  for(int c=0;c<256;++c){Vec mask(slots);std::fill_n(mask.begin()+c*128,128,1.);auto term=m.pm(cols[group*256+c],mask);sum=c?m.add(sum,term):std::move(term);}
  result.push_back(shallow_tile_transpose(m,sum));
 }return result;
}
enum class ColumnPairLayout{query,value,score,av};
inline Cipher column_pack_pair(Math& m,std::array<CipherList,2>& cols,ColumnPairLayout layout){
 int width=layout==ColumnPairLayout::score?128:64;for(auto& c:cols){TORCH_CHECK(c.size()==size_t(width),"Wrong pair column count");column_ready(m,c,1);}
 Cipher result;bool first=true;
 for(int h=0;h<2;++h)for(int c=0;c<width;++c){Vec mask(slots);int offset=layout==ColumnPairLayout::value||layout==ColumnPairLayout::av?h*64:0;
  if(layout==ColumnPairLayout::value){for(int half_index=0;half_index<2;++half_index)std::fill_n(mask.begin()+half_index*half+(offset+c)*128,128,1.);}
  else std::fill_n(mask.begin()+h*half+(offset+c)*128,128,1.);
  auto term=m.pm(cols[h][c],mask);result=first?std::move(term):m.add(result,term);first=false;
 }return shallow_tile_transpose(m,result);
}
inline Cipher column_rotate(Math& m, Cipher x, int tokens) {
 return m.rot(std::move(x), tokens * m.column_token_stride);
}
inline CipherList column_qk(Math& m,CipherList q,CipherList key,bool arion){
 TORCH_CHECK(q.size()==64&&key.size()==64,"QK requires head width 64");column_ready(m,q);column_ready(m,key);CipherList output(128);
 if(!arion){for(int d=0;d<128;++d){CipherList rotated;for(auto& c:key)rotated.push_back(column_rotate(m,c,d));output[d]=column_dot(m,q,rotated,.125);}return output;}
 // ARION Algorithm 1: rho_-12g(d_(12g+b)) = sum_s rho_-12g(q_s)*rho_b(k_s).
 std::vector<CipherList> babies(column_baby);
 for(int b=0;b<column_baby;++b)for(auto& c:key)babies[b].push_back(column_rotate(m,c,b));
 // A giant's Q rotations are used only by that giant; stream this workspace.
 for(int g=0;g<column_giants;++g){CipherList giant;for(auto& c:q)giant.push_back(column_rotate(m,c,-column_baby*g));
  for(int b=0;b<column_baby&&g*column_baby+b<128;++b)output[g*column_baby+b]=column_dot(m,giant,babies[b],.125);
 }
 return output;
}
inline Cipher column_diagonal_sum(Math& m,const CipherList& diagonals,bool rotated){
 TORCH_CHECK(diagonals.size()==128,"Expected 128 attention diagonals");Cipher sum;bool first=true;
 for(int g=0;g<column_giants;++g){auto bucket=diagonals[g*column_baby];for(int b=1;b<column_baby&&g*column_baby+b<128;++b)bucket=m.add(bucket,diagonals[g*column_baby+b]);if(rotated)bucket=column_rotate(m,bucket,g*column_baby);sum=first?std::move(bucket):m.add(sum,bucket);first=false;}return sum;
}
inline CipherList column_av(Math& m,CipherList diagonals,CipherList values,bool already_rotated){
 TORCH_CHECK(diagonals.size()==128&&(values.size()==64||values.size()==32),"AV requires 128 diagonals and 64 real or 32 complex columns");column_ready(m,diagonals);column_ready(m,values);
 if(!already_rotated)for(int d=0;d<128;++d)diagonals[d]=column_rotate(m,diagonals[d],-(d/column_baby*column_baby));
 CipherList output;output.reserve(values.size());
 for(auto& v:values){CipherList baby;for(int b=0;b<column_baby;++b)baby.push_back(column_rotate(m,v,b));Cipher sum;
  for(int g=0;g<column_giants;++g){CipherList a,b;for(int j=0;j<column_baby&&g*column_baby+j<128;++j){a.push_back(diagonals[g*column_baby+j]);b.push_back(baby[j]);}
   auto bucket=column_rotate(m,column_dot(m,a,b),g*column_baby);sum=g?m.add(sum,bucket):std::move(bucket);
  }output.push_back(std::move(sum));
 }return output;
}
}
