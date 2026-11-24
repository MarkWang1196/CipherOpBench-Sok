#include "cipheropbench/published.cuh"
#include "cipheropbench/method_registry.cuh"
#include "cipheropbench/published_coefficients.cuh"
#include "column_numerics.cuh"
#include "research_math.cuh"
#include "scale_aware_chebyshev.cuh"
#include <algorithm>
namespace cipheropbench {
namespace {
using namespace detail;
constexpr std::array<std::array<int,2>,6> pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
int hidden_slot(int row,int feature){return (feature%256/128)*half+row*128+feature%128;}
Cipher raw_plain(Math& m,Cipher& x,const k::FlatVec& values){
 PhantomPlaintext p;double q=m.r.ckks->context->get_context_data(x.chain_index()).parms().coeff_modulus().back().value();
 m.r.ckks->encoder.encode(values,x.chain_index(),q*k::scale/x.scale(),p);Cipher y;m.r.ckks->evaluator.multiply_plain(x,p,y);return y;
}
Cipher raw_add(Math& m,Cipher a,const Cipher& b){TORCH_CHECK(a.chain_index()==b.chain_index()&&std::abs(a.scale()/b.scale()-1.)<1e-10,"Incompatible deferred rescale operands");m.r.ckks->evaluator.add_inplace(a,b);return a;}
Cipher finish_plain(Math& m,Cipher x){m.r.ckks->evaluator.rescale_to_next_inplace(x);x.set_scale(k::scale);return x;}
Cipher block_reduce(Math& m,Cipher x,int width){for(int s=width/2;s;s/=2)x=m.add(x,m.rot(x,s));return x;}
Cipher broadcast_block(Math& m,Cipher x,int width,int rows){for(int s=1;s<rows;s*=2)x=m.add(x,m.rot(x,-width*s));return x;}
Cipher block_head(Math& m,Cipher x,int head,bool value){
 return m.route({x},[=](int j){int r=j%8192/64,f=j%64;return std::tuple{0,head*half+r*128+(value?head*64:0)+f,1.};});
}
// BlockFormer PCBMM, alpha=64, independently within each sample's slots.
CipherList block_linear_native(Math& m,CipherList& a,const LinearWeights& w){
 TORCH_CHECK(w.weight.dim()==2&&w.weight.size(0)==int64_t(64*a.size())&&w.weight.size(1)%64==0,
             "BlockFormer projection requires complete 64-feature blocks");
 auto weights=w.weight.contiguous();auto v=weights.accessor<double,2>();CipherList blocks;
 Vec keep(slots);for(int j=0;j<slots;j+=64)keep[j]=1.;
 for(int b=0;b<w.weight.size(1)/64;++b){Cipher out;bool init=false;
  for(int col=0;col<64;++col){Cipher sum;bool first=true;
   for(int in=0;in<int(a.size());++in){bool used=false;k::FlatVec pt(slots);for(int j=0;j<slots;++j){double c=v[in*64+j%64][b*64+col];pt[j]=make_cuDoubleComplex(c,0.);used|=c!=0.;}if(!used)continue;
    auto term=raw_plain(m,a[in],pt);sum=first?std::move(term):raw_add(m,std::move(sum),term);first=false;
   }if(first)continue;sum=finish_plain(m,std::move(sum));auto term=m.rot(m.pm(block_reduce(m,std::move(sum),64),keep),-col);out=init?m.add(out,term):std::move(term);init=true;
  }blocks.push_back(init?std::move(out):m.zero(a[0]));
 }
 return blocks;
}
CipherList block_linear(Math& m,CipherList x,const LinearWeights& w){
 for(auto& c:x)m.need(c,8);CipherList a;for(int b=0;b<12;++b)a.push_back(m.route(x,[=](int j){int row=j%8192/64,f=b*64+j%64;return std::tuple{f/256,hidden_slot(row,f),1.};}));
 auto blocks=block_linear_native(m,a,w);
 CipherList result;for(int g=0;g<3;++g)result.push_back(m.route(blocks,[=](int j){int row=j%half/128,f=g*256+j/half*128+j%128;return std::tuple{f/64,row*64+f%64,1.};}));add_bias(m,result,w.bias);return result;
}
// BlockFormer SSBMM processes a selected key row, then reduces within alpha.
Cipher block_shift(Math& m,Cipher x,int offset,bool batched){
 if(!batched)return m.rot(std::move(x),offset);
 const int sign=offset<0?-1:1;
 for(int remainder=std::abs(offset),bit=1;remainder;remainder>>=1,bit<<=1)
  if(remainder&1)x=m.rot(std::move(x),sign*bit);
 return x;
}
CipherList block_qk_group(Math& m,const Cipher& a,const Cipher& b,bool batched){
 CipherList out(2);bool init[2]{false,false};
 for(int col=0;col<128;++col){
  Vec mask(slots),keep(slots);
  for(int j=0;j<slots;++j){mask[j]=j%8192/64==col?1.:0.;keep[j]=j%64==0?.125:0.;}
  auto row=broadcast_block(m,block_shift(m,m.pm(b,mask),col*64,batched),64,128);
  auto dot=block_reduce(m,column_product(m,a,row),64);
  auto term=block_shift(m,m.pm(dot,keep),-(col%64),batched);
  int block=col/64;
  out[block]=init[block]?m.add(out[block],term):std::move(term);
  init[block]=true;
 }
 return out;
}
CipherList block_qk_native(Math& m,Cipher q,Cipher key){m.need(q,8);m.need(key,8);CipherList native;
 for(int h=0;h<2;++h){
  auto out=block_qk_group(m,block_head(m,q,h,false),block_head(m,key,h,false),false);
  native.push_back(std::move(out[0]));native.push_back(std::move(out[1]));
 }
 return native;
}
Cipher block_qk(Math& m,Cipher q,Cipher key){auto native=block_qk_native(m,q,key);
 return m.route(native,[](int j){int h=j/half,r=j%half/128,c=(r+j%128)%128;return std::tuple{2*h+c/64,r*64+c%64,1.};});
}
// BlockFormer SRBMM retains the released within-block V transpose.
Cipher block_av_group(Math& m,const Cipher& first,const Cipher& second,const Cipher& v,bool batched){
 Cipher transposed;bool init=false;
 for(int d=0;d<64;++d)for(int lower=0;lower<2;++lower){
  if(!d&&lower)continue;
  Vec mask(slots);
  for(int j=0;j<slots;++j){int r=j%8192/64%64,c=j%64;mask[j]=(lower==0?r-c:c-r)==d?1.:0.;}
  auto term=block_shift(m,m.pm(v,mask),(lower==0?1:-1)*63*d,batched);
  transposed=init?m.add(transposed,term):std::move(term);init=true;
 }
 Cipher out;init=false;
 for(int col=0;col<64;++col)for(int b=0;b<2;++b){
  Vec mask(slots),keep(slots);
  for(int j=0;j<slots;++j){mask[j]=j%8192/64==col+b*64?1.:0.;keep[j]=j%64==0?1.:0.;}
  auto row=broadcast_block(m,block_shift(m,m.pm(transposed,mask),(col+b*64)*64,batched),64,128);
  auto dot=block_reduce(m,column_product(m,b?second:first,row),64);
  auto term=block_shift(m,m.pm(dot,keep),-col,batched);
  out=init?m.add(out,term):std::move(term);init=true;
 }
 return out;
}
Cipher block_av_native(Math& m,CipherList native,Cipher value){for(auto& c:native)m.need(c,4);m.need(value,9);CipherList result;
 for(int h=0;h<2;++h)
  result.push_back(block_av_group(m,native[2*h],native[2*h+1],block_head(m,value,h,true),false));
 return m.route(result,[](int j){int h=j/half,r=j%half/128,c=j%128;if(c/64!=h)return std::tuple{-1,0,0.};return std::tuple{h,r*64+c%64,1.};});
}
Cipher block_av(Math& m,Cipher p,Cipher value){m.need(p,9);CipherList native;
 for(int h=0;h<2;++h)for(int b=0;b<2;++b)native.push_back(m.route({p},[=](int j){int row=j%8192/64,c=b*64+j%64;return std::tuple{0,h*half+row*128+(c-row+128)%128,1.};}));
 return block_av_native(m,std::move(native),value);
}
// EncFormer segment-column projection: complex-paired inputs and BSGS16x16.
CipherList enc_linear(Math& m,CipherList x,const LinearWeights& w){
 CipherList input;Vec mask(slots);std::fill_n(mask.begin(),half,1.);
 for(auto& c:x){m.need(c,12);c=shallow_tile_transpose(m,c);input.push_back(m.add(m.pm(c,mask),thor_imaginary(m,m.pm(m.rot(c,half),mask))));}
 std::vector<CipherList> babies(3);for(int h=0;h<3;++h)for(int b=0;b<16;++b)babies[h].push_back(m.rot(input[h],b*128));
 auto weights=w.weight.contiguous();auto a=weights.accessor<double,2>();CipherList blocks;
 for(int out=0;out<6;++out){Cipher total;bool ready=false;
  for(int g=0;g<16;++g){Cipher bucket;bool init=false;
   for(int b=0;b<16;++b)for(int h=0;h<3;++h){k::FlatVec pt(slots);bool used=false;for(int seg=0;seg<256;++seg){int j=(seg-g*16+256)%256,k=(j+g*16+b)%256;double re=0.,im=0.;if(j<128&&k<128){re=.5*a[h*256+k][out*128+j];im=-.5*a[h*256+128+k][out*128+j];}used|=re!=0||im!=0;std::fill_n(pt.begin()+seg*128,128,make_cuDoubleComplex(re,im));}if(!used)continue;
    auto term=raw_plain(m,babies[h][b],pt);bucket=init?raw_add(m,std::move(bucket),term):std::move(term);init=true;
   }if(!init)continue;auto term=m.rot(bucket,g*16*128);total=ready?raw_add(m,std::move(total),term):std::move(term);ready=true;
  }
  if(ready){m.double_real(total);total=finish_plain(m,std::move(total));}else total=m.zero(input[0]);blocks.push_back(shallow_tile_transpose(m,total));
 }
 CipherList result;for(int g=0;g<3;++g)result.push_back(m.route({blocks[2*g],blocks[2*g+1]},[](int j){return std::tuple{j/half,j%half,1.};}));add_bias(m,result,w.bias);return result;
}
Cipher rot_segment(Math& m,const Cipher& x,int t){t=(t%128+128)%128;if(!t)return x;Vec first(slots),last(slots);for(int j=0;j<slots;++j){first[j]=j%128<128-t?1.:0.;last[j]=1-first[j];}return m.add(m.pm(m.rot(x,t),first),m.pm(m.rot(x,t-128),last));}
std::array<std::pair<int,int>,64> fold_map(){std::array<std::pair<int,int>,64> out;for(auto& v:out)v={-1,-1};for(int j=0;j<8;++j)for(int s=0;s<16;++s){int t=(j*16-s+128)%128;if(t<64&&out[t].first<0)out[t]={j,s};}for(auto v:out)TORCH_CHECK(v.first>=0,"Incomplete EncFormer folded diagonal map");return out;}
Cipher enc_qk(Math& m,Cipher q,Cipher key){m.need(q,12);m.need(key,12);
 auto pack=[&](Cipher x){x=shallow_tile_transpose(m,std::move(x));return m.route({x},[](int j){int seg=j/128,r=j%128;if(seg>=128)return std::tuple{-1,0,0.};int h=seg%2,f=seg/2;return std::tuple{0,h*half+f*128+r,1.};});};q=pack(q);key=pack(key);
 CipherList qb,kb;for(int s=0;s<16;++s)qb.push_back(rot_segment(m,q,s));for(int j=0;j<8;++j)kb.push_back(rot_segment(m,key,j*16));auto map=fold_map();Cipher packed;bool init=false;
 for(int t=0;t<64;++t){auto [j,s]=map[t];auto combo=m.add(kb[j],thor_imaginary(m,kb[(j+4)%8]));auto dot=column_product(m,qb[s],combo),sum=dot;
  for(int d=1;d<64;d*=2)sum=m.add(sum,m.rot(sum,d*256));Vec mask(slots);std::fill_n(mask.begin(),256,.125);auto term=m.rot(m.pm(sum,mask),-t*256);packed=init?m.add(packed,term):std::move(term);init=true;
 }
 auto re=real_part(m,packed),im=imaginary_part(m,packed);
 auto unshift=[&](Cipher a){return m.route({a},[=](int j){int seg=j/128,t=seg/2;if(t>=64)return std::tuple{-1,0,0.};return std::tuple{0,seg*128+(j%128-map[t].second+128)%128,1.};});};
 auto cols=m.route({unshift(re),unshift(im)},[](int j){int h=j/half,d=j%half/128,r=j%128;return std::tuple{d/64,(d%64)*256+h*128+r,1.};});return shallow_tile_transpose(m,std::move(cols));
}
// EncFormer value kernel consumes head-major complex folded probabilities.
Cipher enc_av(Math& m,Cipher p,Cipher value){m.need(p,12);m.need(value,12);
 auto value_columns=shallow_tile_transpose(m,std::move(value));auto v=m.route({value_columns},[](int j){int seg=j/128,h=seg/64,f=seg%64,r=j%128;if(h>=2)return std::tuple{-1,0,0.};return std::tuple{0,h*half+(h*64+f)*128+r,1.};});
 auto score_columns=shallow_tile_transpose(m,std::move(p));auto component=[&](int imag){return m.route({score_columns},[=](int j){int seg=j/128,h=seg/64,t=seg%64,r=j%128;if(h>=2)return std::tuple{-1,0,0.};return std::tuple{0,h*half+((256-t-imag*64)%128)*128+r,1.};});};
 auto fd=m.add(component(0),thor_imaginary(m,component(1)));auto u=m.sub(v,thor_imaginary(m,rot_segment(m,v,64)));CipherList shifted;for(int t=0;t<64;++t)shifted.push_back(rot_segment(m,u,128-t));
 Vec starts(slots);for(int seg=0;seg<256;seg+=64)std::fill_n(starts.begin()+seg*128,128,1.);Cipher total;bool ready=false;
 for(int t=0;t<64;++t){auto bt=m.pm(m.rot(fd,t*128),starts);for(int s=1;s<64;s*=2)bt=m.add(bt,m.rot(bt,-s*128));auto term=column_product(m,shifted[t],bt);total=ready?m.add(total,term):std::move(term);ready=true;
 }
 auto real=real_part(m,total);auto columns=m.route({real},[](int j){int h=j/half,c=j%half/128,row=j%128;if(c/64!=h)return std::tuple{-1,0,0.};return std::tuple{0,(h*64+c%64)*128+row,1.};});return shallow_tile_transpose(m,std::move(columns));
}
Cipher cgf_softmax(Math& m,Cipher x,const k::FlatVec& mask){
 TORCH_CHECK(mask.size()==slots,"CGF requires canonical BERT score mask");m.need(x,14);Vec valid(slots),inv(slots),logn(slots);
 for(int h=0;h<2;++h)for(int row=0;row<128;++row){int n=0;for(int d=0;d<128;++d)n+=cuCreal(mask[h*half+row*128+d])==0;TORCH_CHECK(n>0,"CGF requires a valid key in every row");bool query=cuCreal(mask[h*half+row])==0;
  for(int d=0;d<128;++d){int j=h*half+row*128+d;valid[j]=query&&cuCreal(mask[j])==0?1.:0.;inv[j]=1./n;logn[j]=-std::log(double(n));}
 }
 auto mean=m.pm(m.row_sum(m.pm(x,valid)),inv);auto centered=m.pm(m.sub(x,mean),valid);auto squared=column_product(m,centered,centered);for(auto& v:inv)v*=.5;auto variance=m.pm(m.row_sum(squared),inv);auto z=m.sub(centered,variance);
 auto pt=k::CKKSEncode(logn,m.r.ckks,&z);m.r.ckks->evaluator.add_plain_inplace(z,pt);
 static const ResearchPolynomial exponential{published::cgf_exp_low,published::cgf_exp_high,Vec(published::cgf_exp.begin(),published::cgf_exp.end())};
 auto y=scale_aware_prescaled_activation(m,m.scale(z,1./16),exponential);
 for(int i=0;i<4;++i){if(y.coeff_modulus_size()<=2)column_bounded_refresh(m,y,std::exp2(std::ceil(std::max(0.,published::cgf_exp_high)*(1<<i)/std::log(2.))));y=column_product(m,y,y);}return m.pm(y,valid);
}
}
PublishedOperators::PublishedOperators(Runtime& r,std::string method):TransformerOperators(r),method_(std::move(method)){TORCH_CHECK(is_published_method(method_)&&!r.calibration.enabled,"Published components reject unrelated calibration");}
CipherList PublishedOperators::encrypt_block_batch(const torch::Tensor& input,int levels) const{
 TORCH_CHECK(method_=="blockformer"&&input.dim()==3&&input.size(0)==block_batch_size&&input.size(1)==128&&
             input.size(2)>0&&input.size(2)%64==0,"Expected four BlockFormer samples and complete feature blocks");
 auto cpu=input.to(torch::kCPU).to(torch::kDouble);
 CipherList encrypted;
 for(int f=0;f<cpu.size(2);f+=64){
  auto block=cpu.narrow(2,f,64).contiguous();
  const auto* data=block.data_ptr<double>();
  k::FlatVec packed(slots);
  for(int i=0;i<slots;++i)packed[i]=make_cuDoubleComplex(data[i],0.);
  encrypted.push_back(runtime_.encrypt(std::move(packed),k::chain_idx(levels)));
 }
 return encrypted;
}
torch::Tensor PublishedOperators::decrypt_block_batch(const CipherList& input) const{
 TORCH_CHECK(method_=="blockformer"&&!input.empty(),"Expected BlockFormer ciphertext blocks");
 std::vector<torch::Tensor> values;
 for(const auto& block:input)values.push_back(runtime_.decrypt_slots(block).reshape({block_batch_size,128,64}));
 return torch::cat(values,2);
}
CipherList PublishedOperators::batch_block_linear(CipherList& input,const LinearWeights& w){
 TORCH_CHECK(method_=="blockformer"&&!input.empty()&&w.bias.numel()==w.weight.size(1),"Expected BlockFormer batch projection");
 Math m{runtime_};m.prime_scale=true;
 auto output=block_linear_native(m,input,w);
 for(size_t b=0;b<output.size();++b){
  Vec bias(slots);
  auto values=w.bias.narrow(0,b*64,64).contiguous();
  const auto* data=values.data_ptr<double>();
  for(int i=0;i<slots;i+=64)std::copy_n(data,64,bias.begin()+i);
  auto pt=k::CKKSEncode(std::move(bias),runtime_.ckks,&output[b]);
  runtime_.ckks->evaluator.add_plain_inplace(output[b],pt);
 }
 return output;
}
CipherList PublishedOperators::batch_block_qk(Cipher query,Cipher key){
 TORCH_CHECK(method_=="blockformer","Expected BlockFormer native QK");
 Math m{runtime_};m.prime_scale=true;
 return block_qk_group(m,query,key,true);
}
Cipher PublishedOperators::batch_block_av(const CipherList& probabilities,Cipher value){
 TORCH_CHECK(method_=="blockformer"&&probabilities.size()==2,"Expected BlockFormer native AV");
 Math m{runtime_};m.prime_scale=true;
 return block_av_group(m,probabilities[0],probabilities[1],value,true);
}
CipherList PublishedOperators::batch_block_softmax(CipherList input,const k::FlatVecArray& shift){
 TORCH_CHECK(method_=="blockformer"&&input.size()==2&&shift.size()==input.size(),"Expected BlockFormer score blocks");
 auto& ckks=*runtime_.ckks;
 Math m{runtime_};m.prime_scale=true;
 const double square_input_scale=std::ldexp(1.,block_softmax_working_scale_bits);
 const double coefficient=1./std::sqrt(std::sqrt(block_softmax_divisor));
 for(size_t i=0;i<input.size();++i){
  TORCH_CHECK(input[i].coeff_modulus_size()>=5,"BlockFormer Softmax needs four rescalings");
  auto pt=k::CKKSEncode(shift[i],runtime_.ckks,&input[i]);
  ckks.evaluator.add_plain_inplace(input[i],pt);
  const auto chain=input[i].chain_index();
  const auto prime=ckks.context->get_context_data(chain).parms().coeff_modulus().back().value();
  // Give the two ciphertext squares extra precision before returning to the common scale.
  ckks.encoder.encode(coefficient,chain,prime*square_input_scale/input[i].scale(),pt);
  ckks.evaluator.multiply_plain_inplace(input[i],pt);
  ckks.evaluator.rescale_to_next_inplace(input[i]);
  input[i].set_scale(square_input_scale);
  for(int power=0;power<2;++power){
   Cipher square;
   ckks.evaluator.square(input[i],square);
   ckks.evaluator.relinearize_inplace(square,*ckks.relin_keys);
   ckks.evaluator.rescale_to_next_inplace(square);
   input[i]=std::move(square);
  }
  input[i]=m.scale(std::move(input[i]),1.);
 }
 return input;
}
CipherList PublishedOperators::batch_block_gelu(CipherList input,double linear,double bias){
 TORCH_CHECK(method_=="blockformer"&&!input.empty(),"Expected BlockFormer FFN blocks");
 Math m{runtime_};m.prime_scale=true;
 for(auto& block:input){
  auto square=column_product(m,block,block);
  block=m.addc(m.add(square,m.scale(std::move(block),linear)),bias);
 }
 return input;
}
CipherList PublishedOperators::batch_block_norm(CipherList input,const torch::Tensor& bias){
 TORCH_CHECK(method_=="blockformer"&&input.size()==12&&bias.numel()==768,
             "Expected BlockFormer LayerNorm feature blocks");
 Math m{runtime_};m.prime_scale=true;
 auto sum=input[0];
 for(size_t i=1;i<input.size();++i)sum=m.add(std::move(sum),input[i]);
 Vec start_mask(slots);
 for(int i=0;i<slots;i+=64)start_mask[i]=1./768.;
 auto mean=broadcast_block(m,m.pm(block_reduce(m,std::move(sum),64),start_mask),1,64);
 for(size_t i=0;i<input.size();++i){
  input[i]=m.sub(std::move(input[i]),mean);
  auto beta=bias.narrow(0,i*64,64).repeat({slots/64});
  auto pt=k::CKKSEncode(k::vector_from_tensor<double>(beta),runtime_.ckks,&input[i]);
  runtime_.ckks->evaluator.add_plain_inplace(input[i],pt);
 }
 return input;
}
bool published_scope_supported(const std::string& method,const std::string& scope){if(!is_published_method(method))return true;if(scope=="bootstrap"||scope=="ckks_add"||scope=="ckks_mul")return true;return method=="cgf_softmax"?(scope=="softmax"||scope=="softmax_v"):(scope=="linear"||scope=="qkv"||scope=="qk"||scope=="av"||(method=="blockformer"&&scope=="qk_av"));}
CipherList PublishedOperators::linear(CipherList x,const LinearWeights& w){if(method_=="cgf_softmax")unsupported();TORCH_CHECK(x.size()==3&&w.weight.sizes()==torch::IntArrayRef({768,768})&&w.bias.numel()==768,"Published linear requires common BERT-base shape");Math m{runtime_};m.prime_scale=true;CipherList y;runtime_.record(method_+"_projection_with_canonical_adapters",[&]{y=method_=="blockformer"?block_linear(m,std::move(x),w):enc_linear(m,std::move(x),w);});return y;}
QKV PublishedOperators::project_qkv(CipherList x,const LayerWeights& w){auto q=linear(x,w.q),key=linear(x,w.key),v=linear(std::move(x),w.v);Math m{runtime_};m.prime_scale=true;QKV out;
 for(int p=0;p<6;++p){auto [a,b]=pairs[p];auto map=[=](int j){int row=j%half/128,c=j%128;if(c>=64)return std::tuple{-1,0,0.};int f=(j/half?b:a)*64+c;return std::tuple{f/256,hidden_slot(row,f),1.};};out.q[p]=m.route(q,map);out.key[p]=m.route(key,map);out.v[p]=m.route(v,[=](int j){int row=j%half/128,c=j%128,f=(c<64?a:b)*64+c%64;return std::tuple{f/256,hidden_slot(row,f),1.};});}return out;
}
Cipher PublishedOperators::qk(Cipher q,Cipher key){if(method_=="cgf_softmax")unsupported();Math m{runtime_};m.prime_scale=true;Cipher y;runtime_.record(method_+"_score_with_canonical_adapters",[&]{y=method_=="blockformer"?block_qk(m,q,key):enc_qk(m,q,key);});return y;}
Cipher PublishedOperators::av(Cipher p,Cipher v){if(method_=="cgf_softmax")return TransformerOperators::av(p,v);Math m{runtime_};m.prime_scale=true;Cipher y;runtime_.record(method_+"_value_with_canonical_adapters",[&]{y=method_=="blockformer"?block_av(m,p,v):enc_av(m,p,v);});return y;}
Cipher PublishedOperators::qk_av(Cipher q,Cipher key,Cipher value){if(method_!="blockformer")unsupported();Math m{runtime_};m.prime_scale=true;Cipher y;runtime_.record("blockformer_connected_qk_value_native_blocks",[&]{auto scores=block_qk_native(m,q,key);y=block_av_native(m,std::move(scores),value);});return y;}
Cipher PublishedOperators::softmax(Cipher x,const k::FlatVec& mask){if(method_!="cgf_softmax")unsupported();Math m{runtime_};m.prime_scale=true;Cipher y;runtime_.record("cgf_order2_exp15_k4_with_valid_token_statistics",[&]{y=cgf_softmax(m,x,mask);});return y;}
Cipher PublishedOperators::softmax_v(Cipher x,Cipher v,const k::FlatVec& mask){if(method_!="cgf_softmax")unsupported();return av(softmax(x,mask),v);}
[[noreturn]] void PublishedOperators::unsupported()const{TORCH_CHECK(false,method_," exposes only its published operator scopes; complete FHE classifier not implemented");}
CipherList PublishedOperators::attention(CipherList,const LayerWeights&,torch::Tensor){unsupported();}CipherList PublishedOperators::gelu(CipherList){unsupported();}CipherList PublishedOperators::layer_norm(CipherList,torch::Tensor,torch::Tensor){unsupported();}CipherList PublishedOperators::ffn(CipherList,const LayerWeights&){unsupported();}CipherList PublishedOperators::block(CipherList,const LayerWeights&,torch::Tensor){unsupported();}CipherList PublishedOperators::encoder(CipherList,const std::vector<LayerWeights>&,torch::Tensor){unsupported();}
void register_published_methods(){for(auto name:{"blockformer","encformer","cgf_softmax"})MethodRegistry::instance().add({name,"PhantomBootstrapping_common",std::string(name)=="blockformer"?"blockformer_alpha64_single_input_adapters":std::string(name)=="encformer"?"encformer_scp_complex_bsgs16_fold16_headpair_adapters":"cgf_order2_ls_exp15_k4_masked",std::string(name)=="blockformer"?"cb150bd75cc835be9106632d192d8acdc6d27836":std::string(name)=="encformer"?"a6f8508719876455dbcb4755df58c049696d897c":"f47ecb0daf6c06f5f46ef77d41d6a724b31ebb2a",{}},[name](Runtime& r){return std::make_unique<PublishedOperators>(r,name);});}
}
