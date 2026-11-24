#include "cipheropbench/extended.cuh"
#include "cipheropbench/method_registry.cuh"
#include "extended_matrix.cuh"
#include <fstream>
#include <numbers>
namespace cipheropbench {
namespace {
using detail::Math;using detail::Vec;using detail::half;using detail::slots;
constexpr std::array<std::array<int,2>,6> pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
Vec cheb_coefficients(const std::function<double(double)>& fn,double low,double high,int degree){
  Vec c(degree+1);for(int k=0;k<=degree;++k){for(int j=0;j<=degree;++j){double theta=std::numbers::pi*(j+.5)/(degree+1);c[k]+=fn((low+high)/2+(high-low)/2*std::cos(theta))*std::cos(k*theta);}c[k]*=2./(degree+1);}c[0]/=2;return c;
}
Cipher cheb_series(Math& m,Cipher x,const Vec& c){
  // Affine centering and conjugate projection keep refreshed Chebyshev inputs real.
  if(x.coeff_modulus_size()<=11){x=m.scale(m.addc(x,2.),.125);m.r.refresh(x);m.double_real(x);x=m.addc(m.scale(x,4.),-2.);}
  CipherList basis(c.size());basis[0]=m.addc(m.zero(x),1.);basis[1]=x;
  for(size_t j=2;j<c.size();++j){auto product=j%2?m.mul(basis[j/2],basis[j/2+1]):m.square(basis[j/2]);basis[j]=j%2?m.sub(m.add(product,product),x):m.addc(m.add(product,product),-1.);}
  auto result=m.addc(m.zero(x),c[0]);for(size_t j=1;j<c.size();++j)if(c[j]!=0)result=m.add(result,m.scale(basis[j],c[j]));return result;
}
[[noreturn]] void hybrid_only(){TORCH_CHECK(false,"BLB exposes FHE matrix operators only; its nonlinear/MPC cuts are unavailable in the FHE-only track");}
}
bool extended_scope_supported(const std::string& method,const std::string& scope){
  if(method!="blb")return true;
  return scope=="linear"||scope=="qkv"||scope=="qk"||scope=="av"||scope=="bootstrap"||scope=="ckks_add"||scope=="ckks_mul";
}
ExtendedOperators::ExtendedOperators(Runtime& r,std::string method,std::string statistics):TransformerOperators(r),method_(std::move(method)),statistics_path_(std::move(statistics)){
  TORCH_CHECK(method_=="blb"||method_=="fhe_bert_tiny","Unknown extended method");
  if(statistics_path_.empty())return;
  TORCH_CHECK(method_=="fhe_bert_tiny","Precomputed statistics apply only to FHE-BERT-Tiny");
  std::ifstream in(statistics_path_);std::string magic,task,model,revision;int layers=0,tokens=0;
  in>>magic>>task>>model>>revision>>layers>>tokens;
  TORCH_CHECK(in&&magic=="CIPHEROPBENCH_FHEBERT_STATS_V1"&&layers==12&&tokens==128,"Invalid training-only FHE-BERT statistics header");statistics_.resize(layers);
  for(auto& layer:statistics_)for(auto& norm:layer)for(int kind=0;kind<2;++kind){norm[kind].resize(tokens);for(auto& v:norm[kind]){in>>v;TORCH_CHECK(in&&std::isfinite(v)&&(kind==0||v>0),"Invalid precomputed normalization statistic");}}
  std::string trailing;TORCH_CHECK(!(in>>trailing),"Trailing FHE-BERT statistics fields");
}
CipherList ExtendedOperators::linear(CipherList x,const LinearWeights& w){
  TORCH_CHECK(x.size()==3&&w.weight.sizes()==torch::IntArrayRef({768,768})&&w.bias.numel()==768,"Common Linear requires 128x768 @ 768x768 with bias");Math m{runtime_};CipherList result;
  runtime_.record(method_=="blb"?"blb_spatial_cpmm_bsgs16_with_layout":"fhebert_vecmat_rc_128_tiles_with_layout",[&]{result=method_=="blb"?detail::blb_linear(m,std::move(x),w):detail::tiny_rc_linear(m,std::move(x),w);});return result;
}
QKV ExtendedOperators::project_qkv(CipherList x,const LayerWeights& w){
  Math m{runtime_};CipherList q,key,v;
  runtime_.record(method_=="blb"?"blb_qkv_cpmm":"fhebert_qkv_vecmat_er",[&]{if(method_=="fhe_bert_tiny"){q=detail::tiny_linear(m,x,w.q);key=detail::tiny_linear(m,x,w.key);v=detail::tiny_linear(m,x,w.v);}else{q=linear(x,w.q);key=linear(x,w.key);v=linear(x,w.v);}});QKV out;
  for(int p=0;p<6;++p){auto [a,b]=pairs[p];auto map=[=](int i){int h=i/half,t=(i%half)/128,c=i%128,head=h?b:a;if(c>=64)return std::tuple{-1,0,0.};int f=head*64+c;return std::tuple{f/256,((f/128)%2)*half+t*128+f%128,1.};};
    out.q[p]=m.route(q,map);out.key[p]=m.route(key,map);out.v[p]=m.route(v,[=](int i){int t=(i%half)/128,c=i%128,f=(c<64?a:b)*64+c%64;return std::tuple{f/256,((f/128)%2)*half+t*128+f%128,1.};});
  }return out;
}
Cipher ExtendedOperators::qk(Cipher q,Cipher key){Math m{runtime_};Cipher out;
  runtime_.record(method_=="blb"?"blb_mhp_spatial_reduce_ccmm_bsgs":"fhebert_vecmat_rc_qk",[&]{
    if(method_=="blb"){m.need(key,13);out=m.scale(detail::blb_ccmm(m,q,detail::tile_transpose(m,key)),.125);}else out=detail::tiny_qk(m,q,key);
  });return out;
}
Cipher ExtendedOperators::av(Cipher p,Cipher value){Math m{runtime_};Cipher out;
  runtime_.record(method_=="blb"?"blb_padded_mhp_av_ccmm":"fhebert_vecmat_er_av",[&]{
    if(method_=="fhe_bert_tiny"){m.integer_scale(p,128);out=m.scale(detail::tiny_av(m,p,value),1./128);return;}
    m.integer_scale(p,128);m.integer_scale(value,4);m.need(p,13);m.need(value,13);auto a=m.row_diagonal(p,true);
    auto b=m.route({value},[](int i){int h=i/half,t=(i%half)/128,c=i%128;if(c>=64)return std::tuple{-1,0,0.};return std::tuple{0,h*half+t*128+h*64+c,1.};});
    auto diagonal=detail::blb_ccmm(m,a,b);auto rows=m.row_diagonal(diagonal,true);
    out=m.route({rows},[](int i){int h=i/half,t=(i%half)/128,c=i%128;if((!h&&c>=64)||(h&&c<64))return std::tuple{-1,0,0.};return std::tuple{0,h*half+t*128+c%64,1./512};});
  });return out;
}
CipherList ExtendedOperators::gelu(CipherList x){
  if(method_=="blb")hybrid_only();Math m{runtime_};CipherList out;int stage=calibration_layer_==0?0:1;double bound=stage?17.:13.5;int degree=stage?59:119;
  auto coefficients=cheb_coefficients([](double z){return .5*z*(1+std::erf(z/std::sqrt(2.)));},-bound,bound,degree);
  for(auto& ct:x)runtime_.record("fhebert_source_chebyshev_gelu",[&]{m.need(ct,12);auto normalized=m.scale(ct,1/bound);out.push_back(cheb_series(m,normalized,coefficients));});return out;
}
Cipher ExtendedOperators::softmax(Cipher x,const k::FlatVec& mask){
  if(method_=="blb")hybrid_only();Math m{runtime_};Cipher result;
  runtime_.record("fhebert_taylor6_pow8_chebyshev_reciprocal",[&]{
    m.need(x,12);Vec scaled_mask(slots),inactive(slots);for(int i=0;i<slots;++i){bool valid=cuCreal(mask.at(i))==0.;scaled_mask[i]=valid?1./8:0.;inactive[i]=valid?0.:-1.;}
    auto z=m.pm(x,scaled_mask);auto e=m.poly(z,{1.,1.,.5,1./6,1./24,1./120,1./720});for(int j=0;j<3;++j)e=m.square(e);
    // The release zeroes inactive scores before exp, then subtracts exp(0).
    auto padding=k::CKKSEncode(inactive,runtime_.ckks,&e);runtime_.ckks->evaluator.add_plain_inplace(e,padding);auto sum=m.row_sum(e);
    int stage=calibration_layer_==0?0:1;double low=stage?3.:2.,high=stage?145000.:5000.;int degree=stage?200:119;
    auto normalized=m.addc(m.scale(sum,2/(high-low)),-(high+low)/(high-low));
    auto inverse=cheb_series(m,normalized,cheb_coefficients([](double v){return 1/v;},low,high,degree));result=m.mul(e,inverse);
  });return result;
}
CipherList ExtendedOperators::layer_norm(CipherList x,torch::Tensor weight,torch::Tensor bias){
  if(method_=="blb")hybrid_only();TORCH_CHECK(statistics_.size()==12,"FHE-BERT precomputed normalization needs --fhebert-stats from training data; no exact LayerNorm fallback");
  Math m{runtime_};const auto& values=statistics_.at(calibration_layer_).at(calibration_norm_-1);CipherList out;
  auto packed_weight=k::row_pack_768x1(weight),packed_bias=k::row_pack_768x1(bias);
  runtime_.record("fhebert_training_precomputed_affine_norm",[&]{for(int group=0;group<3;++group){Vec multiplier(slots),offset(slots);for(int i=0;i<slots;++i){int token=(i%half)/128;multiplier[i]=values[1][token]*cuCreal(packed_weight[group][i]);offset[i]=cuCreal(packed_bias[group][i])-values[0][token]*multiplier[i];}
    auto result=m.pm(x.at(group),multiplier);auto p=k::CKKSEncode(offset,runtime_.ckks,&result);runtime_.ckks->evaluator.add_plain_inplace(result,p);out.push_back(std::move(result));}});return out;
}
CipherList ExtendedOperators::ffn(CipherList x,const LayerWeights& w){
  if(method_=="blb")hybrid_only();Math m{runtime_};CipherList out;
  for(int block=0;block<4;++block){LinearWeights up{w.up.weight.slice(1,block*768,(block+1)*768),w.up.bias.slice(0,block*768,(block+1)*768)};
    auto activated=gelu(detail::tiny_linear(m,x,up));LinearWeights down{w.down.weight.slice(0,block*768,(block+1)*768),torch::zeros({768},torch::kDouble)};auto part=linear(activated,down);
    if(block==0)out=std::move(part);else for(int i=0;i<3;++i)out[i]=m.add(out[i],part[i]);
  }
  auto bias=k::row_pack_768x1(w.down.bias);for(int i=0;i<3;++i){auto pt=k::CKKSEncode(bias[i],runtime_.ckks,&out[i]);runtime_.ckks->evaluator.add_plain_inplace(out[i],pt);out[i]=m.add(out[i],x[i]);}return out;
}
CipherList ExtendedOperators::block(CipherList x,const LayerWeights& w,torch::Tensor mask){if(method_=="blb")hybrid_only();TORCH_CHECK(statistics_.size()==12,"FHE-BERT block needs training-only precomputed statistics");return TransformerOperators::block(std::move(x),w,mask);}
CipherList ExtendedOperators::encoder(CipherList x,const std::vector<LayerWeights>& w,torch::Tensor mask){if(method_=="blb")hybrid_only();return TransformerOperators::encoder(std::move(x),w,mask);}
void register_extended_methods(const std::string& statistics){
  auto& registry=MethodRegistry::instance();registry.add({"blb","PhantomBootstrapping_common","blb_spatial_reduce_mhp_bsgs_canonical_boundaries","e696c69bd5b1b2f15b91be0b3b964c0b31304d2b",{}},[](Runtime& r){return std::make_unique<ExtendedOperators>(r,"blb");});
  registry.add({"fhe_bert_tiny","PhantomBootstrapping_common","fhebert_er_rc_tiles128_base_shape_extension","e183da923d2912b9713bee697c2a4f361b318e6d",{}},[statistics](Runtime& r){return std::make_unique<ExtendedOperators>(r,"fhe_bert_tiny",statistics);});
}
}
