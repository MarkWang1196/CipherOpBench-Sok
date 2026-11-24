#include "cipheropbench/research.cuh"
#include "cipheropbench/method_registry.cuh"
#include "column_matrix.cuh"
#include "column_numerics.cuh"
#include "research_columns.cuh"
#include "research_tricycle.cuh"
#include "research_coefficients.cuh"
#include "scale_aware_chebyshev.cuh"
#include <fstream>
namespace cipheropbench {
namespace {
using namespace detail;
constexpr std::array<std::array<int,2>,6> pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
std::vector<AtlasLayer> read_atlas(const std::string& path){
 std::ifstream f(path);std::string magic;int layers=0,tokens=0;f>>magic>>layers>>tokens;
 TORCH_CHECK(f&&(magic=="CIPHEROPBENCH_ATLAS_V1"||magic=="CIPHEROPBENCH_ATLAS_CALIBRATED_V1")&&layers==12&&tokens==128,"Invalid frozen ATLAS profile header: ",path);
 auto finite=[&](double& v){f>>v;TORCH_CHECK(f&&std::isfinite(v),"Nonfinite/truncated ATLAS profile");};
 auto poly=[&](ResearchPolynomial& p){size_t n=0;finite(p.low);finite(p.high);f>>n;TORCH_CHECK(f&&n>=2&&n<=1024&&p.low<p.high,"Invalid ATLAS polynomial");p.coefficients.resize(n);for(auto& c:p.coefficients)finite(c);};
 std::vector<AtlasLayer> result(12);
 for(auto& l:result){f>>l.squares;finite(l.maximum);for(auto& n:l.newton)f>>n;for(auto& n:l.goldschmidt)f>>n;
  TORCH_CHECK(f&&l.squares>=1&&l.squares<=5,"Invalid ATLAS iteration count");for(int i=0;i<2;++i)TORCH_CHECK(l.newton[i]>=1&&l.newton[i]<=(magic=="CIPHEROPBENCH_ATLAS_CALIBRATED_V1"?12:4)&&l.goldschmidt[i]>=1&&l.goldschmidt[i]<=2,"Invalid ATLAS normalization iterations");
  poly(l.exponential);poly(l.gelu);l.inverse_sqrt.resize(l.squares);for(auto& p:l.inverse_sqrt)poly(p);
  for(auto& scale:l.norm_scale){scale.resize(128);for(auto& v:scale){finite(v);TORCH_CHECK(v>0,"Invalid public ATLAS scale");}}
 }std::string rest;TORCH_CHECK(!(f>>rest),"Trailing ATLAS fields");return result;
}
CipherList connected_ensi_norm(Math& m,const CipherList& input,torch::Tensor weight,torch::Tensor bias){
 // Same released sqrt60 and reciprocal60, connected without client decryption.
 // Track public modulus gains and fuse final gain/gamma into the final product.
 CipherList squared;double gain=0.;
 for(const auto& c:input){
  auto result=scale_aware_product(m,{c,1.},{c,1.});
  if(gain)TORCH_CHECK(std::abs(gain-result.gain)<1e-12,"ENSI square gain mismatch");
  gain=result.gain;squared.push_back(result.value);
 }
 auto variance=m.scale(m.hidden_sum(squared),1./(768*gain));
 auto square_root=scale_aware_cheb(m,variance,research_polynomial([](double x){return sqrt(x);},0.,100.,60));
 auto root=m.scale(square_root.value,1./square_root.gain);
 m.need(root,12);
 // Row/half averaging is an identity on the replicated RMS statistic.
 auto sum=m.row_sum(root);root=m.scale(m.add(sum,m.rot(sum,half)),1./256);
 auto inverse=scale_aware_cheb(m,root,research_polynomial([](double x){return 1/x;},.1,10.,60));
 m.need(inverse.value);auto packed=k::row_pack_768x1(weight);CipherList out;
 for(int group=0;group<3;++group){
  auto c=input[group];m.need(c,2);
  size_t chain=std::max(c.chain_index()+1,inverse.value.chain_index());
  double q=m.r.ckks->context->get_context_data(chain).parms().coeff_modulus().back().value();
  Vec factors(slots);for(int i=0;i<slots;++i)factors[i]=cuCreal(packed[group][i])*q/(k::scale*inverse.gain);
  auto weighted=m.pm(c,factors);out.push_back(scale_aware_product(m,{weighted,1.},{inverse.value,1.}).value);
 }
 // Retain BERT bias as an explicit common-graph extension of source RMSNorm.
 add_bias(m,out,bias);return out;
}


Vec valid_mask(const k::FlatVec& mask){TORCH_CHECK(mask.size()==slots,"Mask must contain common slots");Vec valid(slots);for(int i=0;i<slots;++i)valid[i]=cuCreal(mask[i])==0?1.:0.;return valid;}
}
bool is_research_method(const std::string& s){return s=="stip"||s=="tricycle"||s=="ensi"||s=="atlas";}
ResearchOperators::ResearchOperators(Runtime& r,std::string name,const std::string& atlas,bool domains):TransformerOperators(r),method_(std::move(name)),published_domains_(domains){
 TORCH_CHECK(is_research_method(method_),"Unknown research method");TORCH_CHECK(!r.calibration.enabled,"Existing calibration is not a research-method profile");
 if(method_=="atlas")atlas_=read_atlas(atlas.empty()?CIPHEROPBENCH_ATLAS_DEFAULT_PROFILE:atlas);
}
CipherList ResearchOperators::linear(CipherList x,const LinearWeights& w){
 TORCH_CHECK(x.size()==3&&w.weight.sizes()==torch::IntArrayRef({768,768})&&w.bias.numel()==768,"Research Linear requires the common 128x768 @ 768x768 contract");Math m{runtime_};m.prime_scale=method_=="ensi"&&ternary_scale(w)==0.;CipherList out;
 runtime_.record(method_+"_linear_with_canonical_layout",[&]{
  if(method_=="atlas")out=TransformerOperators::linear(x,w);
  else if(method_=="tricycle")out=tricycle_linear(m,x,w);
  else if(method_=="ensi"&&ternary_scale(w)!=0.){runtime_.record("ensi_native_ternary_add_negate_with_dequantization",[&]{out=ensi_bitlinear(m,x,w);});}
  else{runtime_.record(method_=="ensi"?"ensi_dense_common_weights_extension_no_quantization":"stip_dual_head_complex_adaptive_columns",[&]{out=research_column_linear(m,x,w,method_=="stip");});}
 });return out;
}
QKV ResearchOperators::project_qkv(CipherList x,const LayerWeights& w){
 if(method_=="atlas")return TransformerOperators::project_qkv(std::move(x),w);
 auto q=linear(x,w.q),key=linear(x,w.key),v=linear(x,w.v);Math m{runtime_};QKV out;
 for(int p=0;p<6;++p){auto [a,b]=pairs[p];auto map=[=](int i){int h=i/half,t=(i%half)/128,c=i%128,head=h?b:a;if(c>=64)return std::tuple{-1,0,0.};int f=head*64+c;return std::tuple{f/256,((f/128)%2)*half+t*128+f%128,1.};};
  out.q[p]=m.route(q,map);out.key[p]=m.route(key,map);out.v[p]=m.route(v,[=](int i){int t=(i%half)/128,c=i%128,f=(c<64?a:b)*64+c%64;return std::tuple{f/256,((f/128)%2)*half+t*128+f%128,1.};});
 }return out;
}
Cipher ResearchOperators::qk(Cipher q,Cipher key){Math m{runtime_};m.prime_scale=true;Cipher out;runtime_.record(method_+"_qk_with_layout",[&]{
 if(method_=="stip")out=stip_qk(m,q,key);else if(method_=="tricycle")out=tricycle_qk(m,q,key);else if(method_=="ensi")out=ensi_qk(m,q,key);else out=TransformerOperators::qk(q,key);
 });return out;
}
Cipher ResearchOperators::av(Cipher p,Cipher v){Math m{runtime_};m.prime_scale=true;Cipher out;runtime_.record(method_+"_av_with_layout",[&]{
 if(method_=="stip")out=stip_av(m,p,v);else if(method_=="tricycle")out=tricycle_av(m,p,v);else if(method_=="ensi")out=ensi_av(m,p,v);else out=TransformerOperators::av(p,v);
 });return out;
}
CipherList ResearchOperators::affine(CipherList x,torch::Tensor weight,torch::Tensor bias){Math m{runtime_};m.prime_scale=true;auto gamma=k::row_pack_768x1(weight);for(int i=0;i<3;++i){Vec v(slots);for(int j=0;j<slots;++j)v[j]=cuCreal(gamma[i][j]);x[i]=m.pm(x[i],v);}add_bias(m,x,bias);return x;}
CipherList ResearchOperators::gelu(CipherList x){Math m{runtime_};m.prime_scale=published_domains_;CipherList out;
 for(auto& ct:x)runtime_.record(method_+"_activation_source_circuit",[&]{
  if(method_=="atlas"){m.prime_scale=true;out.push_back(scale_aware_prescaled_activation(m,ct,atlas_.at(layer_).gelu));}
  else if(method_=="tricycle"){m.prime_scale=true;out.push_back(scale_aware_prescaled_activation(m,ct,research_polynomial([](double x){return .5*x*(1+erf(x/sqrt(2.)));},-30.,30.,119)));}
  else if(method_=="ensi"){auto sigmoid=research_cheb(m,ct,research_polynomial([](double x){return 1/(1+exp(-x));},-10.,11.,16));out.push_back(m.mul(ct,sigmoid));}
  else{auto sign=research_cheb(m,published_domains_?m.scale(ct,1./16):ct,{-1,1,STIP_RELU_0});sign=research_cheb(m,sign,{-1,1,STIP_RELU_1});out.push_back(m.mul(ct,m.addc(sign,.5)));}
 });return out;
}
Cipher ResearchOperators::softmax(Cipher x,const k::FlatVec& mask){Math m{runtime_};m.prime_scale=method_=="atlas"||method_=="tricycle"||published_domains_;auto valid=valid_mask(mask);Cipher out;
 runtime_.record(method_+"_attention_nonlinear_source_circuit",[&]{
  if(method_=="stip"){out=m.pm(m.exp_euston(m.pm(x,valid),published_domains_?5:1),valid);return;}
  if(method_=="ensi"){auto z=m.addc(m.pm(x,valid),-log(128.));out=m.pm(research_cheb(m,z,research_polynomial([](double v){return 1/(1+exp(-v));},-10.,11.,16)),valid);return;}
  if(method_=="atlas"){
   const auto& l=atlas_.at(layer_);auto z=m.scale(m.addc(x,-l.maximum),1./(1<<l.squares));Vec padding(slots);
   for(int i=0;i<slots;++i)padding[i]=valid[i]?0.:-log(256.);z=m.pm(z,valid);auto pt=k::CKKSEncode(padding,runtime_.ckks,&z);runtime_.ckks->evaluator.add_plain_inplace(z,pt);auto y=research_cheb(m,z,l.exponential);
   for(const auto& p:l.inverse_sqrt){auto denominator=m.row_sum(m.square(y));auto inverse=research_cheb(m,denominator,p);y=m.square(m.mul(y,inverse));}
   out=m.pm(y,valid);return;
  }
  auto z=m.pm(x,valid),mean=m.row_sum(z),squares=m.row_sum(m.square(z));Vec inv_count(slots),alpha0(slots),alpha1(slots);
  for(int i=0;i<slots;++i){int h=i/half,row=(i%half)/128;double n=0;for(int j=0;j<128;++j)n+=valid[h*half+row*128+j];TORCH_CHECK(n>0,"Empty attention row");inv_count[i]=1/n;int head=pairs[pair_][h];double a=1.25,b=layer_==1?.2:.1;
   if(layer_==0&&(head==10||head==11)){a=1.;b=head==10?.13:.14;}
   if(layer_==1&&(head==4||head==6)){a=1.45;b=.115;}if(layer_==1&&head==1){a=1.25;b=.14;}
   if(layer_==2&&head==0){a=.9;b=.165;}if(layer_==2&&head==9){a=1.7;b=.125;}
   alpha0[i]=a*sqrt(2*log(n));alpha1[i]=b*sqrt(2*log(n));
  }
  mean=m.pm(mean,inv_count);auto variance=m.sub(m.pm(squares,inv_count),m.square(mean));auto shift=m.add(mean,m.pm(variance,alpha1));auto pt=k::CKKSEncode(alpha0,runtime_.ckks,&shift);runtime_.ckks->evaluator.add_plain_inplace(shift,pt);
  auto y=m.addc(m.scale(m.sub(z,shift),1./128),1.);for(int j=0;j<7;++j)y=m.square(y);y=m.pm(y,valid);auto denominator=m.row_sum(y);
  double high=layer_==0?640.:(layer_<3?1280.:256.);auto inverse=scale_aware_prescaled_activation(m,denominator,research_polynomial([](double v){return 1/v;},.05,high,119));out=m.mul(y,inverse);
 });pair_=(pair_+1)%6;return out;
}
CipherList ResearchOperators::layer_norm(CipherList x,torch::Tensor weight,torch::Tensor bias){
 TORCH_CHECK(x.size()==3&&weight.numel()==768&&bias.numel()==768,"Research normalization requires common hidden shape");Math m{runtime_};m.prime_scale=true;CipherList out;
 runtime_.record(method_+"_normalization_source_circuit",[&]{
  if(method_=="ensi"){out=connected_ensi_norm(m,x,weight,bias);return;}
  if(method_=="atlas"){const auto& l=atlas_.at(layer_);const auto& factors=l.norm_scale.at(norm_-1);Vec v(slots);for(int i=0;i<slots;++i)v[i]=factors[(i%half)/128];for(auto& c:x)c=m.pm(c,v);}
  CipherList centered,squared;
  auto mean=m.scale(m.hidden_sum(x),1./768);for(auto& c:x)centered.push_back(m.sub(c,mean));
  if(method_=="tricycle"){out=scale_aware_polynomial_norm(m,centered,weight,bias);return;}
  for(auto& c:centered)squared.push_back(m.square(c));auto sum=m.hidden_sum(squared);
  if(method_=="stip"){
   auto root_input=m.addc(m.scale(sum,1./1117),-1.);auto root=m.add(m.addc(m.scale(root_input,.5),1.),m.scale(m.square(root_input),-.125));
   carried_scale_=m.scale(root,sqrt(1117.));carrying_=norm_==1;
   if(norm_==1){auto gamma=k::row_pack_768x1(weight),beta=k::row_pack_768x1(bias);
    // The released kernel encodes gamma*sqrt(width) as one public coefficient.
    for(int g=0;g<3;++g){Vec a(slots),b(slots);for(int i=0;i<slots;++i){a[i]=cuCreal(gamma[g][i])*sqrt(768.);b[i]=cuCreal(beta[g][i]);}out.push_back(m.add(m.pm(centered[g],a),m.pm(carried_scale_,b)));}return;
   }auto inverse=m.inverse(root,2.,4);for(auto& c:centered)out.push_back(m.scale(m.mul(c,inverse),sqrt(768./1117)));out=affine(out,weight,bias);return;
  }
  auto variance=m.scale(sum,1./768);Cipher inverse;
  {
   const auto& l=atlas_.at(layer_);inverse=m.addc(m.scale(variance,-1.29054537e-4),.129054537);
   for(int j=0;j<l.newton[norm_-1];++j)inverse=m.scale(m.mul(inverse,m.addc(m.scale(m.mul(variance,m.square(inverse)),-1.),3.)),.5);
   auto z=m.mul(variance,inverse),h=m.scale(inverse,.5);
   for(int j=0;j<l.goldschmidt[norm_-1];++j){auto correction=m.addc(m.scale(m.mul(z,h),-1.),.5);z=m.add(z,m.mul(z,correction));h=m.add(h,m.mul(h,correction));}inverse=m.add(h,h);
  }
  for(auto& c:centered)out.push_back(m.mul(c,inverse));
  out=affine(out,weight,bias);
 });return out;
}
Cipher ResearchOperators::distance(Cipher q,Cipher key){Math m{runtime_};m.prime_scale=true;m.need(q,13);m.need(key,13);
 auto complex_features=[&](const Cipher& x){auto lo=m.route({x},[](int i){if(i%128>=32)return std::tuple{-1,0,0.};return std::tuple{0,i,1.};});auto hi=m.route({x},[](int i){if(i%128>=32)return std::tuple{-1,0,0.};return std::tuple{0,i+32,1.};});return m.add(lo,thor_imaginary(m,hi));};
 auto a=complex_features(q),b=complex_features(key),conjugate=thor_conjugate(m,b);auto an=m.row_sum(m.mul(a,thor_conjugate(m,a))),bn=m.row_sum(m.mul(b,conjugate));Cipher out;
 for(int d=0;d<128;++d){auto product=m.row_sum(m.mul(a,head_row_rotation(m,conjugate,d)));auto dot=real_part(m,product);auto squared=m.sub(m.add(an,head_row_rotation(m,bn,d)),m.add(dot,dot));Vec at(slots);for(int i=0;i<slots;++i)at[i]=i%128==d?1.:0.;auto term=m.pm(squared,at);out=d?m.add(out,term):std::move(term);}
 return out;
}
Cipher ResearchOperators::gaussian_scores(Cipher q,Cipher key){
 TORCH_CHECK(method_=="stip","Gaussian scores are the STIP attention contract");Math m{runtime_};m.prime_scale=true;
 return m.scale(distance(std::move(q),std::move(key)),-1./16);
}
CipherList ResearchOperators::batch_stip_l2(CipherList query,CipherList key){
 TORCH_CHECK(method_=="stip"&&query.size()==64&&key.size()==64,"STIP RIHP requires 64 grouped query and key columns");
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=256;
 column_ready(m,query,6,9);column_ready(m,key,6,9);
 CipherList q,k,q_conj,k_conj;
 // Algorithm 4 pairs feature columns before taking norms and diagonal dot products.
 for(int j=0;j<32;++j){
  q.push_back(m.add(query[j],thor_imaginary(m,query[j+32])));
  k.push_back(m.add(key[j],thor_imaginary(m,key[j+32])));
  q_conj.push_back(thor_conjugate(m,q.back()));
  k_conj.push_back(thor_conjugate(m,k.back()));
 }
 auto q_norm=column_dot(m,q,q_conj),k_norm=column_dot(m,k,k_conj);
 CipherList out;
 for(int d=0;d<128;++d){
  CipherList shifted;
  for(const auto& c:k_conj)shifted.push_back(column_rotate(m,c,d));
  auto cross=column_dot(m,q,shifted);
  out.push_back(real_part(m,m.sub(m.add(q_norm,column_rotate(m,k_norm,d)),m.add(cross,cross))));
 }
 return out;
}
CipherList ResearchOperators::batch_stip_gk(CipherList diagonals,torch::Tensor valid){
 TORCH_CHECK(method_=="stip"&&diagonals.size()==128&&valid.sizes()==torch::IntArrayRef({32,128}),
             "STIP GK requires 128 diagonals and the masks of 32 inputs");
 Math m{runtime_};m.prime_scale=true;
 auto mask=valid.to(torch::kCPU).to(torch::kBool).contiguous();
 auto keep=mask.accessor<bool,2>();
 for(int d=0;d<128;++d){
  Vec pairs(slots);
  for(int t=0;t<128;++t)for(int lane=0;lane<6*32;++lane)
   pairs[t*256+lane]=keep[lane%32][t]&&keep[lane%32][(t+d)%128]?1.:0.;
  auto z=m.scale(std::move(diagonals[d]),-1./16);
  diagonals[d]=m.pm(m.exp_euston(m.pm(std::move(z),pairs),published_domains_?5:1),pairs);
 }
 return diagonals;
}
CipherList ResearchOperators::batch_stip_av(CipherList diagonals,CipherList value){
 TORCH_CHECK(method_=="stip"&&diagonals.size()==128&&value.size()==64,"STIP RIHP AV requires 128 diagonals and 64 grouped value columns");
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=256;
 column_ready(m,value,6,8);
 CipherList packed;
 for(int j=0;j<32;++j)packed.push_back(m.add(value[j],thor_imaginary(m,value[j+32])));
 auto result=column_av(m,std::move(diagonals),std::move(packed),false);
 for(int j=0;j<32;++j){
  value[j]=real_part(m,result[j]);
  value[j+32]=imaginary_part(m,result[j]);
 }
 return value;
}
#include "stip_batch_encoder.cuh"
CipherList ResearchOperators::attention(CipherList x,const LayerWeights& w,torch::Tensor mask){
 pair_=0;if(method_!="stip")return TransformerOperators::attention(std::move(x),w,mask);
 Math m{runtime_};auto projected=project_qkv(x,w);auto packed=pack_score_mask(mask);CipherList outputs;
 for(int i=0;i<6;++i){auto exponent=gaussian_scores(projected.q[i],projected.key[i]);outputs.push_back(av(softmax(exponent,packed),projected.v[i]));}
 CipherList merged;for(int i=0;i<3;++i)merged.push_back(m.add(outputs[2*i],m.rot(outputs[2*i+1],half)));return linear(merged,w.out);
}
CipherList ResearchOperators::ffn(CipherList x,const LayerWeights& w){Math m{runtime_};CipherList out;
 auto bias=[&](CipherList values,torch::Tensor b){if(method_=="stip"&&carrying_){auto packed=k::row_pack_768x1(b);for(int i=0;i<3;++i){Vec v(slots);for(int j=0;j<slots;++j)v[j]=cuCreal(packed[i][j]);values[i]=m.add(values[i],m.pm(carried_scale_,v));}}else add_bias(m,values,b);return values;};
 for(int block=0;block<4;++block){LinearWeights up{w.up.weight.slice(1,block*768,(block+1)*768),torch::zeros({768},torch::kDouble)};
  auto activated=gelu(bias(linear(x,up),w.up.bias.slice(0,block*768,(block+1)*768)));LinearWeights down{w.down.weight.slice(0,block*768,(block+1)*768),torch::zeros({768},torch::kDouble)};auto part=linear(activated,down);
  if(block==0)out=std::move(part);else for(int i=0;i<3;++i)out[i]=m.add(out[i],part[i]);
 }out=bias(out,w.down.bias);for(int i=0;i<3;++i)out[i]=m.add(out[i],x[i]);return out;
}
CipherList ResearchOperators::block(CipherList x,const LayerWeights& w,torch::Tensor mask){
 Math m{runtime_};pair_=0;carrying_=false;auto y=attention(x,w,mask);for(int i=0;i<3;++i)y[i]=m.add(y[i],x[i]);runtime_.refresh(y);norm_=1;y=layer_norm(y,w.ln1_weight,w.ln1_bias);runtime_.refresh(y);y=ffn(y,w);norm_=2;y=layer_norm(y,w.ln2_weight,w.ln2_bias);runtime_.refresh(y);norm_=1;carrying_=false;return y;
}
CipherList ResearchOperators::encoder(CipherList x,const std::vector<LayerWeights>& weights,torch::Tensor mask){
 for(size_t i=0;i<weights.size();++i){layer_=int(i);x=block(x,weights[i],mask);if(runtime_.layer_completed)runtime_.layer_completed(i+1,weights.size());}layer_=0;return x;
}
void register_research_methods(const std::string& profile,bool domains){
 auto& r=MethodRegistry::instance();
 r.add({"stip","PhantomBootstrapping_common","stip_dhp_amcp_single_sample_rihp_gk_relu_scaled_ln","unversioned_release_sha256_in_source_manifest",{}},[domains](Runtime& x){return std::make_unique<ResearchOperators>(x,"stip","",domains);});
 r.add({"tricycle","PhantomBootstrapping_common","tricycle_crt_tiles_cpmm_bsgs16_ccmm_native_left_right_bsgs_lazy","83c4294238805a9bf295d8ea1ad077fb2f0e6321",{}},[](Runtime& x){return std::make_unique<ResearchOperators>(x,"tricycle");});
 r.add({"ensi","PhantomBootstrapping_common","ensi_columns_universal_vectors_dense_extension_and_explicit_ternary_fastpath","9b6cf7250c937c2dfe7821340dff596700701d35",{}},[](Runtime& x){return std::make_unique<ResearchOperators>(x,"ensi");});
 r.add({"atlas","PhantomBootstrapping_common","atlas_nexus_matrix_with_frozen_layerwise_approximation_profile","039b804bbd871e865ea88b5f411d733957ecc4d0",{}},[profile](Runtime& x){return std::make_unique<ResearchOperators>(x,"atlas",profile);});
}
}
