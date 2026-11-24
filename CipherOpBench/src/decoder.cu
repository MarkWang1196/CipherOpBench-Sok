#include "cipheropbench/decoder.cuh"
#include "research_math.cuh"
#include "column_numerics.cuh"
#include "arion_exp_precision.cuh"
#include "arion_inverse.cuh"
#include <type_traits>
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
#include "cipheropbench/unified_refresh.cuh"
#endif
namespace cipheropbench {
namespace {
using namespace detail;
int stride_for(int width){TORCH_CHECK(width>0&&width<=slots,"Invalid decoder row width");int s=1;while(s<width)s*=2;return s;}
void validate(const DecoderRows& x,bool allow_input45=false){TORCH_CHECK(x.rows>0&&x.width>0&&x.stride==stride_for(x.width)&&x.blocks.size()==size_t((int64_t(x.rows)*x.stride+slots-1)/slots),"Invalid decoder ciphertext layout");for(const auto& c:x.blocks)TORCH_CHECK(c.scale()==k::scale||(allow_input45&&c.scale()==std::ldexp(1.,45)),"This decoder operator requires common40 scale; input45 is limited to ARION pointwise evaluation and client verification");}
Cipher rotate(Math& m,Cipher x,int shift){
 unsigned value=(shift%slots+slots)%slots;
 for(int step=1;value;step*=4,value>>=2)for(unsigned digit=value&3;digit;--digit)x=m.rot(std::move(x),step);return x;
}
// Public routing only: source slot addresses never depend on encrypted contents.
DecoderRows route(Math& m,const CipherList& input,int rows,int width,const std::function<int64_t(int,int)>& source){
 TORCH_CHECK(!input.empty()&&rows>0,"Empty decoder route");DecoderRows out{{},rows,width,stride_for(width)};int count=(int64_t(rows)*out.stride+slots-1)/slots;
 for(int b=0;b<count;++b){std::map<std::pair<int,int>,Vec> groups;
  for(int j=0;j<slots;++j){int64_t pos=int64_t(b)*slots+j;int r=pos/out.stride,c=pos%out.stride;if(r>=rows||c>=width)continue;int64_t s=source(r,c);if(s<0)continue;TORCH_CHECK(s<int64_t(input.size())*slots,"Decoder route source out of range");auto key=std::pair{int(s/slots),int((s%slots-j+slots)%slots)};auto& mask=groups[key];if(mask.empty())mask.resize(slots);mask[j]=1.;}
  Cipher y;bool ready=false;for(auto& [key,mask]:groups){auto z=m.pm(rotate(m,input[key.first],key.second),mask);y=ready?m.add(y,z):std::move(z);ready=true;}out.blocks.push_back(ready?std::move(y):m.zero(input[0]));
 }return out;
}
Cipher spread(Math& m,Cipher x,int width){
 x=rotate(m,std::move(x),1-width);for(int s=1;s<width;s*=2)x=m.add(x,rotate(m,x,s));return x;
}
Cipher reduce(Math& m,Cipher x,int width,bool broadcast){
 for(int s=1;s<width;s*=2)x=m.add(x,rotate(m,x,s));Vec keep(slots);for(int i=0;i<slots;i+=width)keep[i]=1.;x=m.pm(x,keep);
 return broadcast?spread(m,std::move(x),width):x;
}
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
// Two calls to the unchanged shared refresh cancel range-amplified refresh error.
// The final public mask/gain multiplication leaves twelve reusable work moduli.
Cipher public_bounded_restore(Math& m,Cipher input,unsigned amplitude,const Vec& mask={},bool normalized=false){
 TORCH_CHECK(m.r.bootstrapper&&amplitude&&!(amplitude&(amplitude-1)),"Public bounded restore requires bootstrap state and a power-of-two range");
 TORCH_CHECK(input.coeff_modulus_size()>=(normalized?1:2),"Missing range-normalization budget");
 constexpr unsigned gain=unified_phantom::precision_refresh_gain;Cipher result;
 m.r.record("decoder_public_bounded_restore",[&]{
  if(!normalized)input=m.scale(std::move(input),1./amplitude);
  auto original=input;unified_phantom::refresh_drop(*m.r.ckks,original,1);
  auto first=unified_phantom::precision_refresh(*m.r.ckks,*m.r.bootstrapper,input,true,double(amplitude));
  auto large_original=unified_phantom::refresh_integer(*m.r.ckks,std::move(original),amplitude);
  auto reduced=first;unified_phantom::refresh_drop(*m.r.ckks,reduced,1);
  Cipher residual;m.r.ckks->evaluator.sub(large_original,reduced,residual);
  residual=unified_phantom::refresh_integer(*m.r.ckks,std::move(residual),gain);
  auto correction=unified_phantom::precision_refresh(*m.r.ckks,*m.r.bootstrapper,std::move(residual),true);
  auto combined=unified_phantom::refresh_integer(*m.r.ckks,std::move(first),gain);
  m.r.ckks->evaluator.add_inplace(combined,correction);
  Vec coefficients(slots,.5/gain);if(!mask.empty()){TORCH_CHECK(mask.size()==size_t(slots),"Restore mask shape");for(int i=0;i<slots;++i)coefficients[i]*=mask[i];}
  result=m.pm(std::move(combined),coefficients);
  Cipher conjugate;m.r.ckks->evaluator.complex_conjugate(result,*m.r.ckks->galois_keys,conjugate);
  result=m.add(std::move(result),std::move(conjugate));
  TORCH_CHECK(result.coeff_modulus_size()==12&&result.scale()==k::scale,"Public restore output budget");
 },input.chain_index(),&result);
 return result;
}
#endif
Cipher polynomial(Math& m,Cipher x,const char* kind,double lo,double hi,int degree){
 auto p=research_polynomial([=](double z){std::string k=kind;if(k=="silu")return z/(1+exp(-z));if(k=="isqrt")return 1/sqrt(z);if(k=="exp")return exp(z);return 1/z;},lo,hi,degree);
 x=m.addc(m.scale(x,2/(hi-lo)),-(lo+hi)/(hi-lo));
 // Top-chain 51-bit primes need gain restoration at each product, before the next Chebyshev recurrence.
 std::map<int,Cipher> basis;basis[1]=x;
 std::function<Cipher(int)> term=[&](int j)->Cipher{auto found=basis.find(j);if(found!=basis.end())return found->second;
  auto product=column_product(m,term(j/2),term(j%2?j/2+1:j/2));
  auto y=j%2?m.sub(m.add(product,product),x):m.addc(m.add(product,product),-1.);basis.emplace(j,y);return y;
 };
 auto y=m.addc(m.zero(x),p.coefficients[0]);for(int j=1;j<=degree;++j)if(p.coefficients[j]!=0.)y=m.add(y,m.scale(term(j),p.coefficients[j]));return y;
}
}
DecoderRows DecoderOperators::encrypt_rows(torch::Tensor x){
 return encrypt_rows(std::move(x),k::scale);
}
DecoderRows DecoderOperators::encrypt_rows(torch::Tensor x,double input_scale){
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
 TORCH_CHECK(input_scale==k::scale,"Unified decoder input must use the shared profile scale");
 constexpr size_t input_chain=k::boot_level+1;
#else
 TORCH_CHECK(input_scale==k::scale||input_scale==std::ldexp(1.,45),"Decoder client supports explicit40 or45-bit input scale");
 constexpr size_t input_chain=1;
#endif
 TORCH_CHECK(x.dim()==2&&x.size(0)>0,"Decoder input is [rows,width]");x=x.to(torch::kCPU).to(torch::kDouble).contiguous();TORCH_CHECK(torch::isfinite(x).all().item<bool>(),"Nonfinite decoder fixture");DecoderRows y{{},int(x.size(0)),int(x.size(1)),stride_for(x.size(1))};auto a=x.accessor<double,2>();
 for(int b=0;b<(int64_t(y.rows)*y.stride+slots-1)/slots;++b){k::FlatVec values(slots);for(int j=0;j<slots;++j){int64_t p=int64_t(b)*slots+j;int r=p/y.stride,c=p%y.stride;values[j]=make_cuDoubleComplex(r<y.rows&&c<y.width?a[r][c]:0.,0.);}
  if(input_scale==k::scale)y.blocks.push_back(runtime_.encrypt(values,input_chain));
  else{PhantomPlaintext plain;runtime_.ckks->encoder.encode(values,input_scale,plain);Cipher c;if(Runtime::input_encryption_mode()=="public_key")runtime_.ckks->encryptor.encrypt(plain,c);else runtime_.ckks->decryptor.encrypt_symmetric(plain,c);y.blocks.push_back(std::move(c));}
 }return y;
}
torch::Tensor DecoderOperators::decrypt_rows(const DecoderRows& x){validate(x,true);auto y=torch::empty({x.rows,x.width},torch::kDouble);auto a=y.accessor<double,2>();for(size_t b=0;b<x.blocks.size();++b){auto v=runtime_.decrypt_slots(x.blocks[b]).reshape({-1}).contiguous();auto p=v.data_ptr<double>();for(int j=0;j<slots;++j){int64_t pos=int64_t(b)*slots+j;int r=pos/x.stride,c=pos%x.stride;if(r<x.rows&&c<x.width)a[r][c]=p[j];}}return y;}
DecoderRows DecoderOperators::rope(const DecoderRows& x,int start,int d,double theta){
 validate(x);TORCH_CHECK(start>=0&&d>0&&d%2==0&&x.width%d==0&&std::isfinite(theta)&&theta>1,"Invalid decoder RoPE");Math m{runtime_};m.prime_scale=true;auto out=x;
 for(size_t b=0;b<x.blocks.size();++b){Vec c(slots),s0(slots),s1(slots);for(int j=0;j<slots;++j){int64_t p=int64_t(b)*slots+j;int r=p/x.stride,f=p%x.stride;if(r>=x.rows||f>=x.width)continue;int feature=f%d;double a=(start+r)*pow(theta,-double(feature/2*2)/d);c[j]=cos(a);if(feature%2)s1[j]=sin(a);else s0[j]=-sin(a);}
  out.blocks[b]=m.add(m.pm(x.blocks[b],c),m.add(m.pm(rotate(m,x.blocks[b],1),s0),m.pm(rotate(m,x.blocks[b],-1),s1)));
 }return out;
}
DecoderRows DecoderOperators::rmsnorm(const DecoderRows& x,torch::Tensor gamma,double epsilon,int arion_kind){
 validate(x);TORCH_CHECK(x.width==4096&&gamma.numel()==4096&&epsilon>0&&arion_kind>=0&&arion_kind<=2,"Decoder RMSNorm requires hidden4096 and a valid profile");Math m{runtime_};m.prime_scale=true;auto out=x;gamma=gamma.to(torch::kCPU).to(torch::kDouble).contiguous();auto g=gamma.data_ptr<double>();double lo=arion_kind?.01:.5,hi=arion_kind?(arion_kind==1?1.:100.):2.;
 for(size_t b=0;b<x.blocks.size();++b){auto variance=reduce(m,column_product(m,x.blocks[b],x.blocks[b]),x.stride,true);if(!arion_kind)variance=m.scale(variance,1./x.width);variance=m.addc(variance,epsilon*(arion_kind?x.width:1));Vec valid(slots),dummy(slots),weight(slots);
  for(int j=0;j<slots;++j){int r=(int64_t(b)*slots+j)/x.stride;valid[j]=r<x.rows?1.:0.;dummy[j]=r<x.rows?0.:(lo+hi)/2;weight[j]=r<x.rows?g[j%x.stride]*(arion_kind?64.:1.):0.;}
  variance=m.pm(variance,valid);auto pt=k::CKKSEncode(dummy,m.r.ckks,&variance);m.r.ckks->evaluator.add_plain_inplace(variance,pt);
  auto inverse=polynomial(m,variance,"isqrt",lo,hi,31);
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
  variance=public_bounded_restore(m,std::move(variance),arion_kind==2?512:arion_kind==1?4:8);
  inverse=public_bounded_restore(m,std::move(inverse),arion_kind?64:8);
  for(int i=0;i<2;++i){
   auto correction=column_product(m,variance,column_product(m,inverse,inverse));
   m.r.ckks->evaluator.negate_inplace(correction);correction=m.addc(std::move(correction),3.);
   inverse=m.scale(column_product(m,inverse,std::move(correction)),.5);
  }
  for(auto& value:weight)value/=512.;
  auto normalized_output=m.pm(column_product(m,x.blocks[b],inverse),weight);
  out.blocks[b]=public_bounded_restore(m,std::move(normalized_output),512,valid,true);
#else
  for(int i=0;i<2;++i)inverse=m.scale(column_product(m,inverse,m.addc(m.scale(column_product(m,variance,column_product(m,inverse,inverse)),-1.),3.)),.5);
  out.blocks[b]=m.pm(column_product(m,x.blocks[b],inverse),weight);
#endif
 }return out;
}
DecoderRows DecoderOperators::swiglu(const DecoderRows& gate,const DecoderRows& up){
 validate(gate);validate(up);TORCH_CHECK(gate.rows==up.rows&&gate.width==up.width&&gate.width==14336,"Common SwiGLU requires equal branches of width 14336");Math m{runtime_};m.prime_scale=true;auto out=gate;
 for(size_t b=0;b<gate.blocks.size();++b){auto y=column_product(m,polynomial(m,gate.blocks[b],"silu",-10.,10.,127),up.blocks[b]);Vec valid(slots);for(int j=0;j<slots;++j){int64_t pos=int64_t(b)*slots+j;valid[j]=pos/gate.stride<gate.rows&&pos%gate.stride<gate.width?1.:0.;}out.blocks[b]=m.pm(y,valid);}return out;
}
namespace {
DecoderRows arion_pointwise(Runtime& runtime,const DecoderRows& x,bool inverse){
 validate(x,true);TORCH_CHECK(x.width<=8192,"ARION attention vectors support up to8192 keys");
 const double input_scale=x.blocks.front().scale();const bool input45=input_scale==std::ldexp(1.,45);
 TORCH_CHECK(input45||input_scale==k::scale,"ARION nonlinear requires explicit40 or45-bit input scale");
 // shortcut: public-key40 inverse below31 remains unstable; input45 has a separate validated budget.
 for(const auto& c:x.blocks)TORCH_CHECK(c.scale()==input_scale&&(input45?c.coeff_modulus_size()>=size_t(inverse?18:26):inverse?c.coeff_modulus_size()==31:c.coeff_modulus_size()>=25),"ARION nonlinear level/scale budget: input40 exp>=25/inverse31; input45 exp>=26/inverse>=18");
 Math m{runtime};m.prime_scale=true;DecoderRows out{{},x.rows,x.width,x.stride};out.blocks.reserve(x.blocks.size());
 for(size_t b=0;b<x.blocks.size();++b){Vec valid(slots),dummy(slots);for(int j=0;j<slots;++j){int64_t pos=int64_t(b)*slots+j;bool active=pos/x.stride<x.rows&&pos%x.stride<x.width;valid[j]=active;dummy[j]=inverse&&!active?1.:0.;}
  auto argument=x.blocks[b];
  // Explicit server-side normalization consumes one level and stays inside the timing boundary.
  if(input45)argument=m.pm(std::move(argument),Vec(slots,1.));
  if(inverse){auto p=k::CKKSEncode(dummy,runtime.ckks,&argument);runtime.ckks->evaluator.add_plain_inplace(argument,p);}
  if(!inverse){out.blocks.push_back(arion_exp_precision(runtime,std::move(argument),valid));continue;}
  auto y=arion_inverse_polynomial(m,argument);
  if(inverse)for(int i=0;i<2;++i){auto correction=column_product(m,argument,y);runtime.ckks->evaluator.negate_inplace(correction);correction=m.addc(std::move(correction),2.);y=column_product(m,y,std::move(correction));}
  out.blocks.push_back(m.pm(std::move(y),valid));
 }return out;
}
}
DecoderRows DecoderOperators::arion_exp(const DecoderRows& x){return arion_pointwise(runtime_,x,false);}
DecoderRows DecoderOperators::arion_inverse(const DecoderRows& x){return arion_pointwise(runtime_,x,true);}
DecoderRows DecoderOperators::append(const DecoderRows& past,const DecoderRows& incoming){
 validate(incoming);if(!past.rows)return incoming;validate(past);TORCH_CHECK(past.width==incoming.width,"Decoder append width mismatch");Math m{runtime_};m.prime_scale=true;
 int per=slots/past.stride,stable=past.rows/per,tail=past.rows%per;auto out=past;out.rows+=incoming.rows;
 if(!tail){out.blocks.insert(out.blocks.end(),incoming.blocks.begin(),incoming.blocks.end());return out;}
 // Zero-padding invariant lets us add into unused tail slots without masking old ciphertexts.
 auto inserted=route(m,incoming.blocks,tail+incoming.rows,past.width,[&](int r,int c)->int64_t{return r<tail?-1:int64_t(r-tail)*incoming.stride+c;});
 inserted.blocks[0]=m.add(past.blocks[stable],inserted.blocks[0]);out.blocks.resize(stable);out.blocks.insert(out.blocks.end(),inserted.blocks.begin(),inserted.blocks.end());return out;
}
void EncryptedKVCache::append(const DecoderRows& k,const DecoderRows& v,int start){
 static_assert(std::is_nothrow_move_constructible_v<Cipher>,"KV commit requires nonthrowing ciphertext handle moves");
 validate(k);validate(v);TORCH_CHECK(k.rows==v.rows&&k.width==1024&&v.width==1024&&start==length&&int64_t(length)+k.rows<=capacity_,"Invalid KV append, overwrite, or capacity overflow");
 TORCH_CHECK(key.rows==length&&value.rows==length,"Inconsistent KV cache metadata");
 if(length){validate(key);validate(value);TORCH_CHECK(key.width==1024&&value.width==1024,"Invalid cached KV width");}
 const int per=detail::slots/1024,stable=length/per,tail=length%per;
 // Prepare both tails before committing; copying stable history would grow GPU allocation per decode step.
 auto prepare=[&](const DecoderRows& past,const DecoderRows& incoming){
  auto bounded=incoming;for(auto& c:bounded.blocks)if(c.coeff_modulus_size()>size_t(retained_levels_))ops_.runtime_.ckks->evaluator.mod_switch_to_inplace(c,k::chain_idx(retained_levels_));
  if(!tail)return bounded;
  return ops_.append(DecoderRows{{past.blocks.back()},tail,1024,1024},bounded);
 };
 auto next_k=prepare(key,k),next_v=prepare(value,v);
 key.blocks.reserve(stable+next_k.blocks.size());value.blocks.reserve(stable+next_v.blocks.size());
 auto commit=[&](DecoderRows& past,DecoderRows& next){past.blocks.resize(stable);for(auto& c:next.blocks)past.blocks.push_back(std::move(c));past.rows=length+k.rows;past.width=past.stride=1024;};
 commit(key,next_k);commit(value,next_v);length+=k.rows;
}
DecoderRows DecoderOperators::qk(const DecoderRows& q,const DecoderRows& key,int start){
 validate(q);validate(key);TORCH_CHECK(q.width==4096&&key.width==1024&&start>=0&&start+q.rows==key.rows,"GQA requires 32 Q/8 KV heads and contiguous cache");Math m{runtime_};m.prime_scale=true;int rows=q.rows*32;DecoderRows total;
 for(int t=0;t<key.rows;++t){auto k=route(m,key.blocks,rows,128,[&](int r,int c){return int64_t(t)*key.stride+(r%32/4)*128+c;});DecoderRows dot{{},rows,128,128};
  for(size_t b=0;b<q.blocks.size();++b)dot.blocks.push_back(reduce(m,column_product(m,q.blocks[b],k.blocks[b]),128,false));
  auto column=route(m,dot.blocks,rows,key.rows,[&](int r,int c)->int64_t{return c==t?int64_t(r)*128:-1;});
  if(t==0)total=std::move(column);else for(size_t b=0;b<total.blocks.size();++b)total.blocks[b]=m.add(total.blocks[b],column.blocks[b]);
 }for(auto& c:total.blocks)c=m.scale(c,1/sqrt(128.));return total;
}
DecoderRows DecoderOperators::softmax(const DecoderRows& scores,int queries,int start,int keys){
 validate(scores);TORCH_CHECK(scores.rows==queries*32&&scores.width==keys&&start+queries==keys,"Invalid causal decoder Softmax");Math m{runtime_};m.prime_scale=true;auto out=scores;
 for(size_t b=0;b<scores.blocks.size();++b){Vec mask(slots),scale(slots),dummy(slots);for(int j=0;j<slots;++j){int64_t pos=int64_t(b)*slots+j;int r=pos/scores.stride,c=pos%scores.stride;bool active=r<scores.rows;mask[j]=active&&c<keys&&c<=start+r/32?1.:0.;scale[j]=active?1./(start+r/32+1):0.;dummy[j]=active?0.:1.;}
  // Explicit controlled-input contract: exp17[-1,1], inverse15[.5,1.5]+one Newton.
#ifdef CIPHEROPBENCH_UNIFIED_PHANTOM_GPU
  auto bounded=public_bounded_restore(m,scores.blocks[b],4);
  auto e=m.pm(polynomial(m,std::move(bounded),"exp",-1.,1.,17),mask);
  auto sum=m.pm(reduce(m,e,scores.stride,true),scale);
  auto pt=k::CKKSEncode(dummy,m.r.ckks,&sum);m.r.ckks->evaluator.add_plain_inplace(sum,pt);
  sum=public_bounded_restore(m,std::move(sum),8);
  auto inv=polynomial(m,sum,"inverse",.5,1.5,15);
  auto correction=column_product(m,sum,inv);m.r.ckks->evaluator.negate_inplace(correction);correction=m.addc(std::move(correction),2.);
  inv=column_product(m,inv,std::move(correction));
  auto final_scale=scale;for(auto& coefficient:final_scale)coefficient/=4.;
  auto probability=m.pm(column_product(m,e,inv),final_scale);
  out.blocks[b]=public_bounded_restore(m,std::move(probability),4,mask,true);
#else
  auto e=m.pm(polynomial(m,scores.blocks[b],"exp",-1.,1.,17),mask);auto sum=m.pm(reduce(m,e,scores.stride,true),scale);auto pt=k::CKKSEncode(dummy,m.r.ckks,&sum);m.r.ckks->evaluator.add_plain_inplace(sum,pt);auto inv=polynomial(m,sum,"inverse",.5,1.5,15);
  auto correction=m.addc(m.scale(column_product(m,sum,inv),-1.),2.);inv=column_product(m,inv,correction);out.blocks[b]=m.pm(column_product(m,e,inv),scale);
#endif
 }return out;
}
DecoderRows DecoderOperators::av(const DecoderRows& p,const DecoderRows& value,int queries){
 validate(p);validate(value);TORCH_CHECK(p.rows==queries*32&&p.width==value.rows&&value.width==1024,"Invalid rectangular GQA AV");Math m{runtime_};m.prime_scale=true;DecoderRows total;
 for(int t=0;t<value.rows;++t){auto probability=route(m,p.blocks,queries*32,128,[&](int r,int c)->int64_t{return c==0?int64_t(r)*p.stride+t:-1;});for(auto& c:probability.blocks)c=spread(m,std::move(c),128);auto v=route(m,value.blocks,queries*32,128,[&](int r,int c){return int64_t(t)*value.stride+(r%32/4)*128+c;});
  for(size_t b=0;b<v.blocks.size();++b)v.blocks[b]=column_product(m,v.blocks[b],probability.blocks[b]);if(t==0)total=std::move(v);else for(size_t b=0;b<total.blocks.size();++b)total.blocks[b]=m.add(total.blocks[b],v.blocks[b]);
 }total.rows=queries;total.width=4096;total.stride=4096;return total;
}
}
