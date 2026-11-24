#include "cipheropbench/column_methods.cuh"
#include "cipheropbench/method_registry.cuh"
#include "column_matrix.cuh"
#include "column_numerics.cuh"
#include "scale_aware_chebyshev.cuh"
#include "cipheropbench/calibrated_math.hpp"
#include <fstream>
#include <iostream>
namespace cipheropbench {
namespace {
using namespace detail;
constexpr std::array<std::array<int,2>,6> column_pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
constexpr double moai_shifts[]{7.5,9.9,13.6,13.3,9.5,8.,10.3,9.,9.,9.,11.,7.};
const ResearchPolynomial& arion_poly(const char* kind,double lo,double hi,int degree){
 static std::map<std::tuple<std::string,double,double,int>,ResearchPolynomial> cache;auto key=std::tuple{std::string(kind),lo,hi,degree};auto found=cache.find(key);if(found!=cache.end())return found->second;
 return cache.emplace(key,research_polynomial([=](double x){std::string k=kind;if(k=="exp")return exp(x);if(k=="inverse")return 1/x;if(k=="isqrt")return 1/sqrt(x);return .5*x*(1+tanh(sqrt(2/std::numbers::pi)*(x+.044715*x*x*x)));},lo,hi,degree)).first->second;
}
bool arion_exp_rebased(){const char* value=std::getenv("CIPHEROPBENCH_ARION_EXP_REBASE");return value&&std::string(value)=="1";}
bool arion_batch_fold_gain(){const char* value=std::getenv("CIPHEROPBENCH_ARION_GAIN_FOLDED");return value&&std::string(value)=="1";}
std::pair<double,double> arion_refresh_bounds(int layer,int head){
 const char* file=std::getenv("CIPHEROPBENCH_ARION_REFRESH_PROFILE");
 if(!file)return {};
 static const auto bounds=[&]{
  std::ifstream in(file);std::string magic,task,model,revision;int layers=0,heads=0;
  in>>magic>>task>>model>>revision>>layers>>heads;
  TORCH_CHECK(in&&magic=="CIPHEROPBENCH_ARION_BATCH_REFRESH_V1"&&layers==12&&heads==12,"Invalid ARION refresh profile");
  std::array<std::array<std::pair<double,double>,12>,12> result;
  for(auto& row:result)for(auto& [denominator,inverse]:row){
   in>>denominator>>inverse;
   TORCH_CHECK(in&&std::isfinite(denominator)&&std::isfinite(inverse)&&denominator>=1&&inverse>=1&&
               std::max(denominator,inverse)<=std::exp2(50.),"Invalid ARION refresh bounds");
  }
  TORCH_CHECK(!(in>>magic),"Unexpected trailing ARION refresh bounds");
  return result;
 }();
 return bounds.at(layer).at(head);
}
Cipher arion_activation_masked(Math& m,Cipher input,const ResearchPolynomial& polynomial,Vec mask){
 auto normalized=scale_aware_activation_input(m,std::move(input),polynomial);
 auto result=scale_aware_cheb_normalized(m,std::move(normalized),polynomial.coefficients);
 for(auto& value:mask)value/=result.gain;
 return m.pm(std::move(result.value),mask);
}
const ResearchPolynomial& arion_exp_working_basis(const ColumnProfile& profile,bool batch=false){
 const auto& source=arion_poly("exp",profile.exp_low,profile.exp_high,63);
 if(!arion_exp_rebased())return source;
 TORCH_CHECK(profile.exp_low==-60.&&profile.exp_high==15.,"Working basis requires the released ARION exponential");
 const auto rebase=[&](double low,double high){
  auto evaluate=[&](double value){long double x=(2*value-source.low-source.high)/(source.high-source.low),b1=0,b2=0;for(int j=int(source.coefficients.size())-1;j>=1;--j){auto b=2*x*b1-b2+source.coefficients[j];b2=b1;b1=b;}return double(x*b1-b2+source.coefficients[0]);};
  auto result=research_polynomial(evaluate,low,high,63);
  for(int i=0;i<1024;++i){double x=low+(high-low)*i/1023,t=(2*x-result.low-result.high)/(result.high-result.low),b1=0,b2=0;for(int j=63;j>=1;--j){double b=2*t*b1-b2+result.coefficients[j];b2=b1;b1=b;}TORCH_CHECK(std::abs(t*b1-b2+result.coefficients[0]-evaluate(x))<1e-7,"ARION basis transform changed its source polynomial");}
  return result;
 };
 if(batch){
  static const auto result=rebase(ColumnOperators::arion_batch_exp_low,ColumnOperators::arion_batch_exp_high);
  return result;
 }
 static const auto result=rebase(-20.,4.);
 return result;
}
constexpr double moai_gelu_coefficients[]{3.18006986e-24,5.70792114e-22,3.97205561e-20,1.31854608e-18,1.64153184e-17,-2.33052347e-16,-9.78309547e-15,-6.72238500e-14,1.43093357e-12,2.41129634e-11,-4.00991558e-11,-3.06661368e-09,-1.00479838e-08,2.05368974e-07,1.25666834e-06,-7.76703686e-06,-6.75419265e-05,1.62401656e-04,1.97100905e-03,-1.70511673e-03,-3.22621248e-02,7.22135066e-03,3.39374355e-01,4.92938360e-01,1.21149468e-02};
constexpr double moai_gelu_bias=moai_gelu_coefficients[24];
constexpr double moai_gelu_core_radius=10., moai_gelu_dep=4./27.;
template<class Value>
Value moai_gelu_gate(Value x){
 Value gate=x*0.;
 for(int i=0;i<24;++i)gate=gate*x+moai_gelu_coefficients[i];
 return gate;
}
Cipher column_activation(Math& m,Cipher x,bool arion,const ColumnProfile& profile=ColumnProfile{}){
 if(arion){const auto& p=arion_poly("gelu",profile.gelu_low,profile.gelu_high,255);return research_cheb(m,x,p);}
 static const auto p=research_polynomial([](double x){return x*moai_gelu_gate(x)+moai_gelu_bias;},-20,10,24);
 return research_cheb(m,x,p);
}
Cipher moai_inverse_exact_gains(Math& m,Cipher x){
 if(x.coeff_modulus_size()<16)column_bounded_refresh(m,x,2.);
 m.r.ckks->evaluator.negate_inplace(x);ScaledPolynomialCipher a{m.addc(std::move(x),1.),1.},y{m.addc(a.value,1.),1.};
 auto bound=[](double x){return std::exp2(std::ceil(std::log2(std::max(x,1.))));};
 for(int i=0;i<16;++i){
  if(i==5){column_bounded_refresh(m,a.value,bound(a.gain));column_bounded_refresh(m,y.value,bound(64*y.gain));}
  a=scale_aware_product(m,a,a);
  y=scale_aware_product(m,y,{m.addc(a.value,a.gain),a.gain});
 }
 TORCH_CHECK(std::isfinite(y.gain)&&y.gain>0,"Invalid public reciprocal gain");
 if(y.value.coeff_modulus_size()<=5)column_bounded_refresh(m,y.value,bound(131072*y.gain));
 return m.scale(std::move(y.value),1/y.gain);
}
Cipher column_reciprocal(Math& m,Cipher sum,bool arion,const ColumnProfile& profile=ColumnProfile{},
                         std::pair<double,double> refresh_bounds={}){
 if(!arion)return moai_inverse_exact_gains(m,m.addc(sum,1e-5));
 const auto [fitted_sum,fitted_inverse]=refresh_bounds;
 if(fitted_sum&&sum.coeff_modulus_size()<=13)column_bounded_refresh(m,sum,fitted_sum);
 const auto& p=arion_poly("inverse",profile.inverse_low,profile.inverse_high,255);auto y=scale_aware_prescaled_activation(m,sum,p);
 if(std::getenv("CIPHEROPBENCH_ARION_STAGE_DIAGNOSTIC"))
  std::cerr<<"ARION diagnostic initial_inverse max_abs="<<m.r.decrypt_slots(y).abs().max().item<double>()<<'\n';
 const double sum_bound=fitted_sum?fitted_sum:std::exp2(std::ceil(std::log2(std::max(profile.inverse_high,1.))));
 const double inverse_bound=fitted_inverse?fitted_inverse:std::exp2(std::ceil(std::log2(std::max(2/profile.inverse_low,1.))));
 for(int i=0;i<profile.inverse_iterations;++i){
  if(profile.inverse_iterations>2&&std::min(sum.coeff_modulus_size(),y.coeff_modulus_size())<=3){
   // V2 can produce large reciprocals; use public power-of-two bounds before refreshing.
   TORCH_CHECK(std::isfinite(sum_bound)&&std::isfinite(inverse_bound),"ARION reciprocal refresh bounds overflow");
   column_bounded_refresh(m,sum,sum_bound);column_bounded_refresh(m,y,inverse_bound);
  }
  auto product=column_product(m,sum,y);m.r.ckks->evaluator.negate_inplace(product);y=column_product(m,y,m.addc(product,2.));
 }if(y.coeff_modulus_size()<=5)column_bounded_refresh(m,y,inverse_bound);return y;
}
// Mask uses the canonical diagonal convention [head,query,offset].
Vec column_valid(const k::FlatVec& mask,int head,int diagonal,int giant){
 Vec out(slots);for(int i=0;i<slots;++i){int t=(i%128-giant+128)%128;out[i]=cuCreal(mask[head*half+t*128+diagonal])==0&&cuCreal(mask[head*half+t])==0?1.:0.;}return out;
}
Cipher column_denominator(Math& m,const CipherList& e,bool rotated,const k::FlatVec& mask,int h,const Vec* batch_valid=nullptr){
 auto sum=column_diagonal_sum(m,e,rotated);Vec dummy(slots);for(int j=0;j<slots;++j)dummy[j]=batch_valid?1.-(*batch_valid)[j]:(cuCreal(mask[h*half+j%128])==0?0.:1.);
 auto pt=k::CKKSEncode(dummy,m.r.ckks,&sum);m.r.ckks->evaluator.add_plain_inplace(sum,pt);return sum;
}
CipherList column_exponentials(Math& m,CipherList scores,const k::FlatVec& mask,int h,bool arion,int layer,const ColumnProfile& profile=ColumnProfile{},const Vec* batch_valid=nullptr){
 TORCH_CHECK(scores.size()==128&&(batch_valid?batch_valid->size()==slots:mask.size()==slots),
             "Column attention requires 128 diagonals and a slot mask");
 auto diagonal_valid=[&](int diagonal,int giant){
   if(!batch_valid)return column_valid(mask,h,diagonal,giant);
   Vec valid(slots);
   const int stride=m.column_token_stride;
   for(int i=0;i<slots;++i){
     const int query=(i-giant*stride+slots)%slots;
     const int key=(query+diagonal*stride)%slots;
     valid[i]=(*batch_valid)[query]*(*batch_valid)[key];
   }
   return valid;
 };
 Cipher shift;CipherList rotated_shift;
 if(arion){
  // Released code computes moments over valid keys, with ConstantValue=2.08.
  Vec inv(slots);
  if(batch_valid){
   std::vector<int> valid_keys(m.column_token_stride);
   for(int i=0;i<slots;++i)valid_keys[i%m.column_token_stride]+=(*batch_valid)[i]!=0;
   for(int i=0;i<slots;++i){
    const int count=valid_keys[i%m.column_token_stride];
    inv[i]=count?1./count:0.;
   }
  }else for(int t=0;t<128;++t){double n=0;for(int d=0;d<128;++d)n+=cuCreal(mask[h*half+t*128+d])==0;for(int i=t;i<slots;i+=128){inv[i]=n?1/n:0.;}}
  CipherList masked;for(int d=0;d<128;++d)masked.push_back(m.pm(scores[d],diagonal_valid(d,d/column_baby*column_baby)));
  auto mean=m.pm(column_diagonal_sum(m,masked,true),inv);masked.clear();CipherList rotated_mean;for(int g=0;g<column_giants;++g)rotated_mean.push_back(column_rotate(m,mean,-g*column_baby));CipherList centered_squared;centered_squared.reserve(128);
  for(int d=0;d<128;++d){int g=d/column_baby*column_baby;auto z=m.sub(scores[d],rotated_mean[d/column_baby]);z=m.pm(z,diagonal_valid(d,g));centered_squared.push_back(m.square(z));}
  auto variance=m.pm(column_diagonal_sum(m,centered_squared,true),inv);
  shift=m.add(mean,m.addc(m.scale(variance,.1*profile.statmax),1.25*profile.statmax));
  for(int g=0;g<column_giants;++g)rotated_shift.push_back(column_rotate(m,shift,-g*column_baby));
 }
 const auto& exponential=arion_exp_working_basis(profile,batch_valid!=nullptr);
 if(arion&&!batch_valid&&arion_exp_rebased()){
  // Reuse replicated slots for independent columns; the polynomial commutes with these encrypted permutations.
  for(int d=0;d<128;++d){int g=d/column_baby*column_baby;auto valid=diagonal_valid(d,g);
   scores[d]=m.rot(m.pm(m.sub(scores[d],rotated_shift[d/column_baby]),valid),g);
  }
  std::array<CipherList,2> duplicate{scores,scores};auto packed=column_pack_pair(m,duplicate,ColumnPairLayout::score);
  auto values=scale_aware_prescaled_activation(m,std::move(packed),exponential);
  Vec valid(slots);for(int i=0;i<slots;++i){int t=(i%half)/128,d=i%128;valid[i]=cuCreal(mask[h*half+t*128+d])==0&&cuCreal(mask[h*half+t])==0?1.:0.;}
  double bound=0.;for(double c:exponential.coefficients)bound+=std::abs(c);
  bound=std::exp2(std::ceil(std::log2(std::max(bound,1.))));TORCH_CHECK(std::isfinite(bound),"ARION exponential refresh bound overflow");
  column_bounded_refresh(m,values,bound);
  values=m.pm(std::move(values),valid);
  auto result=pair_columns(m,std::move(values),0,128,false,16);
  for(int d=0;d<128;++d)result[d]=m.rot(result[d],-(d/column_baby*column_baby));
  return result;
 }
 for(int d=0;d<128;++d){int g=arion?d/column_baby*column_baby:0;
  auto z=arion?m.sub(scores[d],rotated_shift[d/column_baby]):m.addc(scores[d],-moai_shifts[layer]);
  auto valid=diagonal_valid(d,g);
  if(arion){
   if(arion_exp_rebased())z=m.pm(std::move(z),valid);
   if(batch_valid&&arion_batch_fold_gain())scores[d]=arion_activation_masked(m,std::move(z),exponential,std::move(valid));
   else scores[d]=m.pm(scale_aware_prescaled_activation(m,z,exponential),valid);
  }
  else{z=m.addc(m.scale(z,1./128),1.);if(z.coeff_modulus_size()<=8)column_bounded_refresh(m,z,2.);ScaledPolynomialCipher e{std::move(z),1.};for(int j=0;j<7;++j)e=scale_aware_product(m,e,e);for(auto& v:valid)v/=e.gain;scores[d]=m.pm(std::move(e.value),valid);}
 }return scores;
}
CipherList column_softmax(Math& m,CipherList s,const k::FlatVec& mask,int h,bool arion,int layer,const ColumnProfile& profile=ColumnProfile{}){
 auto e=column_exponentials(m,std::move(s),mask,h,arion,layer,profile);auto inv=column_reciprocal(m,column_denominator(m,e,arion,mask,h),arion,profile);
 CipherList rotated_inverse;if(arion)for(int g=0;g<column_giants;++g)rotated_inverse.push_back(m.rot(inv,-g*column_baby));
 for(int d=0;d<128;++d)e[d]=column_product(m,e[d],arion?rotated_inverse[d/column_baby]:inv);return e;
}
CipherList column_norm(Math& m,CipherList x,torch::Tensor gamma,torch::Tensor beta,bool arion,int kind,const ColumnProfile& profile=ColumnProfile{},const Vec* valid=nullptr,const calibration::Layer* calibrated=nullptr){
 TORCH_CHECK(x.size()==768&&gamma.numel()==768&&beta.numel()==768,"Column LayerNorm requires width 768");
 auto sum=x[0];for(int i=1;i<768;++i)sum=m.add(sum,x[i]);auto mean=m.scale(sum,1./768);Cipher variance;
 for(int i=0;i<768;++i){x[i]=m.sub(x[i],mean);auto square=m.square(x[i]);variance=i?m.add(variance,square):std::move(square);}
 variance=m.scale(variance,kind==1&&!calibrated?1.:1./768);Cipher inverse;
 if(calibrated){
  Math cal{m.r};
  variance=cal.addc(std::move(variance),m.r.calibration.epsilon);
  Vec padding(slots);
  for(int i=0;i<slots;++i)padding[i]=(1-(*valid)[i])*calibrated->low.at(kind-1);
  auto pt=k::CKKSEncode(padding,m.r.ckks,&variance);
  m.r.ckks->evaluator.add_plain_inplace(variance,pt);
  inverse=calibration::inverse_sqrt(cal,std::move(variance),*calibrated,kind,m.r.calibration.method);
 }else if(arion){const auto& first=arion_poly("isqrt",profile.norm1_low,profile.norm1_high,31);const auto& second=arion_poly("isqrt",profile.norm2_low,profile.norm2_high,31);
  if(valid){Vec dummy(slots);for(int j=0;j<slots;++j)dummy[j]=(1-(*valid)[j])*(kind==1?(profile.norm1_low+profile.norm1_high)/2:(profile.norm2_low+profile.norm2_high)/2);auto pt=k::CKKSEncode(dummy,m.r.ckks,&variance);m.r.ckks->evaluator.add_plain_inplace(variance,pt);}
  inverse=scale_aware_prescaled_activation(m,variance,kind==1?first:second);
  for(int j=0;j<2;++j)inverse=m.scale(column_product(m,inverse,m.addc(m.scale(column_product(m,variance,column_product(m,inverse,inverse)),-1.),3.)),.5);
 }else if(valid){
  const double divisor=kind==1?1024.:16., root_scale=std::sqrt(divisor);
  const double estimate_bound=kind==1?64.:8.;
  // Keep the same Newton and Goldschmidt polynomial under a public change of scale.
  auto unit=m.scale(variance,1./divisor);
  auto estimate=m.addc(m.scale(unit,-1.29054537e-4*divisor*root_scale),.129054537*root_scale);
  auto keep=[&](Cipher& value,double bound){if(value.coeff_modulus_size()<=4)column_bounded_refresh(m,value,bound);};
  for(int j=0;j<4;++j){
   keep(unit,1.);keep(estimate,estimate_bound);
   auto product=column_product(m,column_product(m,unit,estimate),estimate);
   auto correction=m.addc(m.scale(std::move(product),-.5),1.5);
   estimate=column_product(m,std::move(estimate),std::move(correction));
  }
  keep(estimate,estimate_bound);
  auto root=column_product(m,unit,estimate), half=m.scale(estimate,.5);
  for(int j=0;j<2;++j){
   keep(root,2.);keep(half,estimate_bound);
   auto correction=m.addc(m.scale(column_product(m,root,half),-1.),.5);
   root=m.add(root,column_product(m,root,correction));
   half=m.add(half,column_product(m,half,correction));
  }
  inverse=m.scale(std::move(half),2./root_scale);
  if(inverse.coeff_modulus_size()<=5)column_bounded_refresh(m,inverse,2.);
 }else{
  inverse=m.addc(m.scale(variance,-1.29054537e-4),.129054537);
  for(int j=0;j<4;++j)inverse=m.scale(m.mul(inverse,m.addc(m.scale(m.mul(variance,m.square(inverse)),-1.),3.)),.5);
  auto z=m.mul(variance,inverse),half_inverse=m.scale(inverse,.5);
  for(int j=0;j<2;++j){auto correction=m.addc(m.scale(m.mul(z,half_inverse),-1.),.5);z=m.add(z,m.mul(z,correction));half_inverse=m.add(half_inverse,m.mul(half_inverse,correction));}inverse=m.add(half_inverse,half_inverse);
 }
 // Refresh a shared inverse once before the column products copy it.
 m.need(inverse,5);
 for(int i=0;i<768;++i)x[i]=m.addc(m.scale(m.mul(x[i],inverse),gamma[i].item<double>()*(kind==1&&!calibrated?sqrt(768.):1.)),beta[i].item<double>());
 return x;
}

Vec native_token_mask(torch::Tensor mask){Vec out(slots);auto valid=mask[0].contiguous();for(int i=0;i<slots;++i)out[i]=valid[i%128].item<bool>()?1.:0.;return out;}
Vec batch_token_mask(const torch::Tensor& valid){
 auto cpu=valid.to(torch::kCPU).to(torch::kBool).contiguous();
 auto mask=cpu.accessor<bool,2>();Vec out(slots);
 constexpr int stride=ColumnOperators::batch_capacity;
 for(int t=0;t<128;++t)for(int b=0;b<cpu.size(0);++b)out[t*stride+b]=mask[b][t];
 return out;
}
void mask_native_columns(Math& m,CipherList& x,const Vec& valid){for(auto& c:x)c=m.pm(c,valid);}
// Preserve the single-request path's timing instrumentation.
void column_stage(Math& m,const char* name,const std::function<void()>& evaluate){
 if(m.column_token_stride==1)evaluate();
 else m.r.record(std::string("batch.")+name,evaluate);
}
bool arion_stage_diagnostic(Math& m){return m.column_token_stride>1&&std::getenv("CIPHEROPBENCH_ARION_STAGE_DIAGNOSTIC");}
void arion_observe(Math& m,const char* stage,const CipherList& values,double bound){
 if(!arion_stage_diagnostic(m))return;
 double maximum=0.;
 for(const auto& c:values){
  auto observed=m.r.decrypt_slots(c);
  TORCH_CHECK(observed.isfinite().all().item<bool>(),"Nonfinite ARION diagnostic at ",stage);
  maximum=std::max(maximum,observed.abs().max().item<double>());
 }
 std::cerr<<"ARION diagnostic "<<stage<<" max_abs="<<maximum<<'\n';
 TORCH_CHECK(maximum<=bound,"ARION diagnostic exceeded public envelope at ",stage);
}
CipherList moai_batch_softmax(Runtime& r,CipherList scores,const Vec& valid,const calibration::Layer& profile){
 Math m{r};m.prime_scale=true;
 const bool diagnostic=std::getenv("CIPHEROPBENCH_MOAI_BATCH_DIAGNOSTIC")!=nullptr;
 auto observe=[&](const char* stage,const Cipher& value,int step,double gain=1.){
  if(!diagnostic)return;
  const auto plain=r.decrypt_slots(value)/gain;
  std::cerr<<"MOAI diagnostic "<<stage<<' '<<step<<" levels="<<value.coeff_modulus_size()
           <<" min="<<plain.min().item<double>()<<" max="<<plain.max().item<double>()<<'\n';
 };
 const int squarings=profile.exp_squarings;
 TORCH_CHECK(squarings>0&&squarings<=20,"Invalid MOAI exponential depth");
 for(int d=0;d<128;++d){
  Vec pair(slots);
  for(int i=0;i<slots;++i)pair[i]=valid[i]*valid[(i+d*ColumnOperators::batch_capacity)%slots];
  ScaledPolynomialCipher value{calibration::softmax_base(m,std::move(scores[d]),profile,"moai"),1.};
  if(d==0)observe("base",value.value,d);
  if(value.value.coeff_modulus_size()<=size_t(squarings+2))column_bounded_refresh(m,value.value,2.);
  for(int j=0;j<squarings;++j){
   // Refresh before the last squares make low-probability terms comparable to rescale noise.
   if(value.value.coeff_modulus_size()<=6){
    value.value=m.scale(std::move(value.value),1./value.gain);value.gain=1.;
    column_bounded_refresh(m,value.value,2.);
   }
   value=scale_aware_product(m,value,value);
   if(d==0&&(j%3==0||j+1==squarings))observe("exp",value.value,j+1,value.gain);
  }
  for(auto& item:pair)item/=value.gain;
  scores[d]=m.pm(std::move(value.value),pair);
 }
 auto denominator=column_denominator(m,scores,false,{},0,&valid);
 observe("denominator",denominator,0);
 if(denominator.coeff_modulus_size()<=5)column_bounded_refresh(m,denominator,2.);
 observe("denominator_ready",denominator,0);
 ScaledPolynomialCipher error{m.addc(m.scale(m.addc(denominator,1e-5),-1.),1.),1.};
 // MOAI uses one Goldschmidt reciprocal per diagonal row sum. Keep it scaled across bootstraps.
 constexpr double divisor=32768.;
 ScaledPolynomialCipher inverse{m.scale(m.addc(error.value,1.),1./divisor),1.};
 auto refresh=[&](ScaledPolynomialCipher& value,double bound){
  if(value.value.coeff_modulus_size()>4)return;
  value.value=m.scale(std::move(value.value),1./value.gain);
  value.gain=1.;column_bounded_refresh(m,value.value,bound);
 };
 for(int j=0;j<16;++j){
  refresh(error,2.);
  refresh(inverse,std::max(1.,std::ldexp(1.,j+1)/divisor));
  error=scale_aware_product(m,error,error);
  inverse=scale_aware_product(m,inverse,{m.addc(error.value,error.gain),error.gain});
  if(j%4==0||j==15){observe("reciprocal_error",error.value,j+1,error.gain);observe("scaled_inverse",inverse.value,j+1,inverse.gain);}
 }
 refresh(inverse,4.);
 for(auto& score:scores){
  auto probability=scale_aware_product(m,{std::move(score),1.},inverse);
  score=m.scale(std::move(probability.value),divisor/probability.gain);
 }
 observe("probability",scores[0],0);
 return scores;
}
CipherList calibrated_batch_softmax(Runtime& r,CipherList scores,const Vec& valid,const calibration::Layer& profile){
 constexpr int stride=ColumnOperators::batch_capacity;
 Math m{r};
 const auto& method=r.calibration.method;
 if(method=="moai")return moai_batch_softmax(r,std::move(scores),valid,profile);
 for(int d=0;d<128;++d){
  Vec pair(slots);
  for(int i=0;i<slots;++i)pair[i]=valid[i]*valid[(i+d*stride)%slots];
  scores[d]=m.pm(calibration::softmax_base(m,std::move(scores[d]),profile,method),pair);
 }
 auto normalize=[&](double maximum){
  auto sum=column_diagonal_sum(m,scores,false);
  Vec padding(slots);
  for(int i=0;i<slots;++i)padding[i]=(1-valid[i])*maximum;
  auto pt=k::CKKSEncode(padding,r.ckks,&sum);
  r.ckks->evaluator.add_plain_inplace(sum,pt);
  auto inverse=calibration::scaled_reciprocal(m,std::move(sum),maximum);
  for(auto& score:scores){score=m.mul(std::move(score),inverse);m.integer_scale(score,int64_t(128/maximum));}
 };
 normalize(128.);
 const int squarings=method=="euston"?profile.euston_softmax_squarings:profile.exp_squarings;
 for(int j=0;j<squarings;++j){
  for(auto& score:scores)score=m.square(std::move(score));
  normalize(1.);
 }
 return scores;
}
CipherList native_column_attention(Math& m,CipherList& input,const LayerWeights& w,torch::Tensor mask,bool arion,int layer,const ColumnProfile& profile=ColumnProfile{},const Vec* batch_valid=nullptr,const calibration::Layer* calibrated=nullptr){
 auto packed_mask=batch_valid?k::FlatVec{}:TransformerOperators::pack_score_mask(mask);
 auto valid=batch_valid?*batch_valid:native_token_mask(mask);
 const bool euston=calibrated&&m.r.calibration.method=="euston";
 CipherList heads;
 for(int h=0;h<12;++h){
  auto slice=[&](const LinearWeights& a){
   return LinearWeights{a.weight.slice(1,h*64,(h+1)*64),a.bias.slice(0,h*64,(h+1)*64)};
  };
  CipherList q,key,v,scores,result;
  column_stage(m,"qkv",[&]{
   q=column_cpmm(m,input,slice(w.q));
   key=column_cpmm(m,input,slice(w.key));
   v=column_cpmm(m,input,slice(w.v));
   mask_native_columns(m,q,valid);mask_native_columns(m,key,valid);mask_native_columns(m,v,valid);
  });
  column_stage(m,"qk",[&]{
   if(euston){column_ready(m,q);column_ready(m,key);scores=LiteratureOperators(m.r,"euston").batch_qk(q,key);}
   else scores=column_qk(m,q,key,arion);
  });
  q.clear();key.clear();
  if(arion){
   CipherList exp;
   Cipher inverse;
   arion_observe(m,"qk",scores,512.);
   column_stage(m,"softmax",[&]{
    exp=column_exponentials(m,std::move(scores),packed_mask,0,true,layer,profile,batch_valid);
    inverse=column_reciprocal(m,column_denominator(m,exp,true,packed_mask,0,batch_valid),true,profile,
                              batch_valid?arion_refresh_bounds(layer,h):std::pair<double,double>{});
   });
   arion_observe(m,"exp",exp,4*std::exp(profile.exp_high));
   if(arion_stage_diagnostic(m))arion_observe(m,"reciprocal",{inverse},4/profile.inverse_low);
   column_stage(m,"av",[&]{result=column_av(m,std::move(exp),v,true);});
   column_stage(m,"softmax",[&]{for(auto& c:result)c=column_product(m,c,inverse);});
   arion_observe(m,"normalized_av",result,512.);
  }else if(calibrated){
   column_stage(m,"softmax",[&]{scores=calibrated_batch_softmax(m.r,std::move(scores),valid,*calibrated);});
   column_stage(m,"av",[&]{
    if(euston){column_ready(m,scores);column_ready(m,v);result=LiteratureOperators(m.r,"euston").batch_av(scores,v);}
    else result=column_av(m,std::move(scores),std::move(v),false);
   });
  }else result=column_av(m,column_softmax(m,std::move(scores),packed_mask,0,false,layer),v,false);
  column_stage(m,"attention_ready",[&]{column_ready(m,result);});
  heads.insert(heads.end(),std::make_move_iterator(result.begin()),std::make_move_iterator(result.end()));
 }
 CipherList out;
 column_stage(m,"wo",[&]{
  out=column_cpmm(m,heads,w.out);
  mask_native_columns(m,out,valid);
 });
 return out;
}
CipherList native_column_ffn(Math& m,CipherList& input,const LayerWeights& w,bool arion,const Vec* valid=nullptr,const ColumnProfile& profile=ColumnProfile{},const calibration::Layer* calibrated=nullptr){
 CipherList sum(768);
 for(int tile=0;tile<48;++tile){
  const int lo=tile*64,hi=lo+64;
  LinearWeights up{w.up.weight.slice(1,lo,hi),w.up.bias.slice(0,lo,hi)};
  CipherList values;
  column_stage(m,"ffn_up",[&]{
   values=column_cpmm(m,input,up);
   if(valid)mask_native_columns(m,values,*valid);
  });
  column_stage(m,"activation",[&]{
   if(arion&&valid&&m.column_token_stride>1&&arion_batch_fold_gain()){
    const auto& polynomial=arion_poly("gelu",profile.gelu_low,profile.gelu_high,255);
    for(auto& c:values)c=arion_activation_masked(m,std::move(c),polynomial,*valid);
    return;
   }
   Math cal{m.r};
   for(auto& c:values)c=calibrated?calibration::gelu(cal,std::move(c),*calibrated,m.r.calibration.method):column_activation(m,c,arion,profile);
   if(valid)mask_native_columns(m,values,*valid);
  });
  if(arion&&tile==0)arion_observe(m,"gelu_first_tile",values,1024.);
  column_stage(m,"ffn_down",[&]{
   for(int output=0;output<768;output+=64){
    LinearWeights down{w.down.weight.slice(0,lo,hi).slice(1,output,output+64),torch::zeros(64,w.down.bias.options())};
    auto part=column_cpmm(m,values,down);
    for(int j=0;j<64;++j)sum[output+j]=tile?m.add(sum[output+j],part[j]):std::move(part[j]);
   }
  });
 }
 column_stage(m,"ffn_residual",[&]{
  for(int i=0;i<768;++i)sum[i]=m.add(m.addc(sum[i],w.down.bias[i].item<double>()),input[i]);
  if(valid)mask_native_columns(m,sum,*valid);
 });
 return sum;
}
CipherList native_column_block(Math& m,CipherList input,const LayerWeights& w,torch::Tensor mask,bool arion,int layer,const ColumnProfile& profile=ColumnProfile{},const Vec* batch_valid=nullptr,const calibration::Layer* calibrated=nullptr){
 auto valid=batch_valid?*batch_valid:native_token_mask(mask);
 column_stage(m,"input_mask",[&]{mask_native_columns(m,input,valid);});
 auto out=native_column_attention(m,input,w,mask,arion,layer,profile,batch_valid,calibrated);
 column_stage(m,"attention_residual",[&]{
  for(int i=0;i<768;++i)out[i]=m.add(out[i],input[i]);
 });
 if(arion)arion_observe(m,"attention_residual",out,1024.);
 input.clear();
 column_stage(m,"ln1",[&]{
  out=column_norm(m,std::move(out),w.ln1_weight,w.ln1_bias,arion,1,profile,&valid,calibrated);
  mask_native_columns(m,out,valid);column_ready(m,out);
 });
 if(arion)arion_observe(m,"ln1",out,1024.);
 out=native_column_ffn(m,out,w,arion,&valid,profile,calibrated);
 if(arion)arion_observe(m,"ffn_residual",out,1024.);
 column_stage(m,"ln2",[&]{
  out=column_norm(m,std::move(out),w.ln2_weight,w.ln2_bias,arion,2,profile,&valid,calibrated);
  mask_native_columns(m,out,valid);column_ready(m,out);
 });
 if(arion)arion_observe(m,"ln2",out,1024.);
 return out;
}
}
ColumnOperators::ColumnOperators(Runtime& r,std::string method,std::string profile):TransformerOperators(r),method_(std::move(method)){
 TORCH_CHECK(is_column_method(method_)||(method_=="euston"&&r.calibration.enabled),"Unknown column method");
 TORCH_CHECK(!r.calibration.enabled||(method_!="arion"&&r.calibration.method==method_),"Column calibration does not match the method");
 if(!profile.empty()){TORCH_CHECK(method_=="arion"&&!r.calibration.enabled,"ARION range profile requires ARION without legacy calibration");std::ifstream in(profile);std::string magic,task,model,revision;int layers;in>>magic>>task>>model>>revision>>layers;TORCH_CHECK(in&&(magic=="CIPHEROPBENCH_ARION_TRAIN_RANGES_V1"||magic=="CIPHEROPBENCH_ARION_TRAIN_RANGES_V2")&&layers==12,"Invalid train ARION profile header");for(auto& p:profiles_){in>>p.statmax>>p.exp_low>>p.exp_high>>p.inverse_low>>p.inverse_high>>p.gelu_low>>p.gelu_high>>p.norm1_low>>p.norm1_high>>p.norm2_low>>p.norm2_high;if(magic=="CIPHEROPBENCH_ARION_TRAIN_RANGES_V2")in>>p.inverse_iterations;TORCH_CHECK(in&&p.inverse_iterations>=2&&p.inverse_iterations<=32&&std::isfinite(p.statmax)&&p.exp_low<p.exp_high&&p.inverse_low>0&&p.inverse_low<p.inverse_high&&p.gelu_low<p.gelu_high&&p.norm1_low>0&&p.norm1_low<p.norm1_high&&p.norm2_low>0&&p.norm2_low<p.norm2_high,"Invalid ARION intervals");}std::string trailing;TORCH_CHECK(!(in>>trailing),"Trailing ARION profile fields");}
}
CipherList ColumnOperators::linear(CipherList x,const LinearWeights& w){
 TORCH_CHECK(x.size()==3&&w.weight.sizes()==torch::IntArrayRef({768,768})&&w.bias.numel()==768,"Column linear requires common 128x768 @ 768x768");Math m{runtime_};m.prime_scale=true;CipherList out;
 runtime_.record(method_+"_column_cpmm_with_canonical_adapters",[&]{auto input=hidden_columns(m,std::move(x));out=column_hidden(m,column_cpmm(m,input,w));});return out;
}
QKV ColumnOperators::project_qkv(CipherList x,const LayerWeights& w){
 Math m{runtime_};m.prime_scale=true;auto input=hidden_columns(m,std::move(x));QKV out;
 for(int p=0;p<6;++p){auto [a,b]=column_pairs[p];auto project=[&](const LinearWeights& weights,bool value){
   auto slice=[&](int h){return LinearWeights{weights.weight.slice(1,h*64,(h+1)*64),weights.bias.slice(0,h*64,(h+1)*64)};};
   std::array<CipherList,2> result{column_cpmm(m,input,slice(a)),column_cpmm(m,input,slice(b))};return column_pack_pair(m,result,value?ColumnPairLayout::value:ColumnPairLayout::query);
  };out.q[p]=project(w.q,false);out.key[p]=project(w.key,false);out.v[p]=project(w.v,true);
 }return out;
}
Cipher ColumnOperators::qk(Cipher q,Cipher key){Math m{runtime_};m.prime_scale=true;std::array<CipherList,2> out;
 for(int h=0;h<2;++h){auto a=pair_columns(m,q,h,64),b=pair_columns(m,key,h,64);out[h]=column_qk(m,a,b,method_=="arion");if(method_=="arion")for(int d=0;d<128;++d)out[h][d]=m.rot(out[h][d],d/column_baby*column_baby);}
 return column_pack_pair(m,out,ColumnPairLayout::score);
}
Cipher ColumnOperators::av(Cipher p,Cipher v){Math m{runtime_};m.prime_scale=true;std::array<CipherList,2> out;
 for(int h=0;h<2;++h){auto s=pair_columns(m,p,h,128);if(method_=="arion")for(int d=0;d<128;++d)s[d]=m.rot(s[d],-(d/column_baby*column_baby));out[h]=column_av(m,std::move(s),pair_columns(m,v,h,64,true),method_=="arion");}
 return column_pack_pair(m,out,ColumnPairLayout::av);
}
Cipher ColumnOperators::softmax(Cipher s,const k::FlatVec& mask){
 if(runtime_.calibration.enabled)return TransformerOperators::softmax(std::move(s),mask);
 Math m{runtime_};m.prime_scale=true;std::array<CipherList,2> out;bool arion=method_=="arion";
 for(int h=0;h<2;++h){auto d=pair_columns(m,s,h,128,false,16);if(arion)for(int j=0;j<128;++j)d[j]=m.rot(d[j],-(j/column_baby*column_baby));out[h]=column_softmax(m,std::move(d),mask,h,arion,layer_,profiles_[layer_]);if(arion)for(int j=0;j<128;++j)out[h][j]=m.rot(out[h][j],j/column_baby*column_baby);}
 return column_pack_pair(m,out,ColumnPairLayout::score);
}
Cipher ColumnOperators::softmax_v(Cipher s,Cipher v,const k::FlatVec& mask){
 if(method_=="moai"||runtime_.calibration.enabled)return av(softmax(std::move(s),mask),std::move(v));
 Math m{runtime_};m.prime_scale=true;std::array<CipherList,2> out;
 for(int h=0;h<2;++h){auto d=pair_columns(m,s,h,128);for(int j=0;j<128;++j)d[j]=m.rot(d[j],-(j/column_baby*column_baby));auto e=column_exponentials(m,std::move(d),mask,h,true,layer_,profiles_[layer_]);
  auto inv=column_reciprocal(m,column_denominator(m,e,true,mask,h),true,profiles_[layer_]);out[h]=column_av(m,std::move(e),pair_columns(m,v,h,64,true),true);for(auto& c:out[h])c=column_product(m,c,inv);
 }return column_pack_pair(m,out,ColumnPairLayout::av);
}
CipherList ColumnOperators::attention(CipherList x,const LayerWeights& w,torch::Tensor mask){
 if(runtime_.calibration.enabled)return TransformerOperators::attention(std::move(x),w,mask);
 Math m{runtime_};m.prime_scale=true;CipherList out;
 runtime_.record(method_+"_connected_column_attention_including_adapters",[&]{auto input=hidden_columns(m,std::move(x));out=column_hidden(m,native_column_attention(m,input,w,mask,method_=="arion",layer_,profiles_[layer_]));});return out;
}
CipherList ColumnOperators::gelu(CipherList x){if(runtime_.calibration.enabled)return TransformerOperators::gelu(std::move(x));Math m{runtime_};m.prime_scale=true;for(auto& c:x)c=column_activation(m,c,method_=="arion",profiles_[layer_]);return x;}
CipherList ColumnOperators::layer_norm(CipherList x,torch::Tensor w,torch::Tensor b){if(runtime_.calibration.enabled)return TransformerOperators::layer_norm(std::move(x),w,b);Math m{runtime_};m.prime_scale=true;return column_hidden(m,column_norm(m,hidden_columns(m,std::move(x)),w,b,method_=="arion",norm_,profiles_[layer_]));}
CipherList ColumnOperators::ffn(CipherList x,const LayerWeights& w){
 if(runtime_.calibration.enabled)return TransformerOperators::ffn(std::move(x),w);
 Math m{runtime_};m.prime_scale=true;auto input=hidden_columns(m,std::move(x));return column_hidden(m,native_column_ffn(m,input,w,method_=="arion",nullptr,profiles_[layer_]));
}
CipherList ColumnOperators::block(CipherList x,const LayerWeights& w,torch::Tensor mask){
 Math m{runtime_};m.prime_scale=true;
 if(!runtime_.calibration.enabled)return column_hidden(m,native_column_block(m,hidden_columns(m,std::move(x)),w,mask,method_=="arion",layer_,profiles_[layer_]));
 calibration_layer_=layer_;auto y=attention(x,w,mask);for(int i=0;i<3;++i)y[i]=m.add(y[i],x[i]);runtime_.refresh(y);norm_=calibration_norm_=1;y=layer_norm(y,w.ln1_weight,w.ln1_bias);runtime_.refresh(y);y=ffn(y,w);norm_=calibration_norm_=2;y=layer_norm(y,w.ln2_weight,w.ln2_bias);runtime_.refresh(y);norm_=calibration_norm_=1;return y;
}
CipherList ColumnOperators::encoder(CipherList x,const std::vector<LayerWeights>& w,torch::Tensor mask){
 if(runtime_.calibration.enabled){for(size_t i=0;i<w.size();++i){layer_=calibration_layer_=int(i);x=block(std::move(x),w[i],mask);if(runtime_.layer_completed)runtime_.layer_completed(i+1,w.size());}layer_=calibration_layer_=0;return x;}
 Math m{runtime_};m.prime_scale=true;auto columns=hidden_columns(m,std::move(x));
 // Preserve the native column state across all sublayers and encoder layers.
 for(size_t i=0;i<w.size();++i){columns=native_column_block(m,std::move(columns),w[i],mask,method_=="arion",int(i),profiles_[i]);if(runtime_.layer_completed)runtime_.layer_completed(i+1,w.size());}return column_hidden(m,std::move(columns));
}
CipherList ColumnOperators::encrypt_batch(torch::Tensor hidden) const {
 return encrypt_batch(std::move(hidden),k::boot_level+1);
}

CipherList ColumnOperators::encrypt_batch(torch::Tensor hidden,size_t chain) const {
 TORCH_CHECK(hidden.dim()==3&&hidden.size(0)>0&&hidden.size(0)<=batch_capacity&&
             hidden.size(1)==128&&hidden.size(2)>0&&hidden.size(2)<=3072,"Expected [batch,128,features]");
 auto cpu=hidden.to(torch::kCPU).to(torch::kDouble).contiguous();
 auto values=cpu.accessor<double,3>();
 CipherList result;
 result.reserve(cpu.size(2));
 for(int f=0;f<cpu.size(2);++f){
  k::FlatVec column(slots);
  for(int t=0;t<128;++t)for(int b=0;b<cpu.size(0);++b)
    column[t*batch_capacity+b]=make_cuDoubleComplex(values[b][t][f],0.);
  result.push_back(runtime_.encrypt(std::move(column),chain));
 }
 return result;
}

torch::Tensor ColumnOperators::decrypt_batch(const CipherList& hidden,int active_inputs) const {
 TORCH_CHECK(!hidden.empty()&&active_inputs>0&&active_inputs<=batch_capacity,
             "Invalid ARION batch output geometry");
 auto output=torch::empty({active_inputs,128,int64_t(hidden.size())},torch::kDouble);
 for(size_t f=0;f<hidden.size();++f)
  output.select(2,f).copy_(runtime_.decrypt_slots(hidden[f]).reshape({128,batch_capacity})
                          .narrow(1,0,active_inputs).transpose(0,1));
 return output;
}

CipherList ColumnOperators::batch_linear(CipherList hidden,const LinearWeights& weights){
 TORCH_CHECK(method_=="arion"&&hidden.size()==size_t(weights.weight.size(0))&&!runtime_.calibration.enabled,
             "Expected uncalibrated ARION batch columns");
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 return column_cpmm(m,hidden,weights);
}

CipherList ColumnOperators::batch_qk(CipherList query,CipherList key){
 TORCH_CHECK(is_column_method(method_),"ARION or MOAI batch layout required");
 const bool arion=method_=="arion";
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 auto scores=column_qk(m,std::move(query),std::move(key),arion);
 for(int d=0;arion&&d<128;++d)
  scores[d]=column_rotate(m,std::move(scores[d]),d/column_baby*column_baby);
 return scores;
}

CipherList ColumnOperators::batch_av(CipherList attention,CipherList value){
 TORCH_CHECK(is_column_method(method_),"ARION or MOAI batch layout required");
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 return column_av(m,std::move(attention),std::move(value),method_=="arion");
}

CipherList ColumnOperators::batch_softmax(CipherList scores,torch::Tensor valid){
 TORCH_CHECK(method_=="moai"&&runtime_.calibration.enabled&&runtime_.calibration.method=="moai",
             "MOAI batch Softmax requires a training calibration");
 TORCH_CHECK(scores.size()==128&&valid.dim()==2&&valid.size(1)==128&&
             valid.size(0)>0&&valid.size(0)<=batch_capacity,"Invalid Softmax batch geometry");
 return calibrated_batch_softmax(runtime_,std::move(scores),batch_token_mask(valid),runtime_.calibration.layers.at(layer_));
}

CipherList ColumnOperators::batch_layer_norm(CipherList hidden,torch::Tensor weight,torch::Tensor bias,
                                            torch::Tensor valid,int kind){
 TORCH_CHECK(is_column_method(method_)&&!runtime_.calibration.enabled&&(kind==1||kind==2),
             "Expected ARION or MOAI column LayerNorm 1 or 2");
 TORCH_CHECK(valid.dim()==2&&valid.size(0)>0&&valid.size(0)<=batch_capacity&&valid.size(1)==128,
             "Invalid LayerNorm batch geometry");
 auto mask=batch_token_mask(valid);
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 auto out=column_norm(m,std::move(hidden),weight,bias,method_=="arion",kind,profiles_.at(layer_),&mask);
 mask_native_columns(m,out,mask);
 return out;
}

std::pair<double,double> ColumnOperators::batch_arion_gelu_domain() const {
 TORCH_CHECK(method_=="arion"&&!runtime_.calibration.enabled,"Expected ARION public GELU profile");
 const auto& profile=profiles_.at(layer_);
 return {profile.gelu_low,profile.gelu_high};
}

Cipher ColumnOperators::batch_arion_gelu(Cipher input,const std::vector<double>& slot_mask){
 TORCH_CHECK(slot_mask.size()==slots,"ARION GELU requires a full batch slot mask");
 const auto [low,high]=batch_arion_gelu_domain();
 const auto& polynomial=arion_poly("gelu",low,high,255);
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 return arion_activation_masked(m,std::move(input),polynomial,slot_mask);
}

torch::Tensor ColumnOperators::moai_batch_gelu_reference(const torch::Tensor& input) const {
 TORCH_CHECK(method_=="moai"&&!runtime_.calibration.enabled,"Expected MOAI batch GELU");
 auto unit=input/moai_gelu_core_radius;
 return input*moai_gelu_gate((unit-moai_gelu_dep*unit.pow(3))*moai_gelu_core_radius)+moai_gelu_bias;
}

Cipher ColumnOperators::batch_moai_gelu(Cipher input,const std::vector<double>& slot_mask){
 TORCH_CHECK(method_=="moai"&&!runtime_.calibration.enabled&&slot_mask.size()==slots,
             "MOAI GELU requires a full batch slot mask");
 // Factor the released degree-24 GELU and extend only its bounded degree-23 gate.
 static const auto gate=research_polynomial([](double z){return moai_gelu_gate(z*moai_gelu_core_radius);},-1.,1.,23);
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 auto unit=m.scale(std::move(input),1./moai_gelu_core_radius);
 if(unit.coeff_modulus_size()<=7)column_bounded_refresh(m,unit,4.);
 auto cubic=column_product(m,column_product(m,unit,unit),unit);
 auto mapped=m.sub(unit,m.scale(std::move(cubic),moai_gelu_dep));
 if(mapped.coeff_modulus_size()<=8)column_bounded_refresh(m,mapped,2.);
 auto value=scale_aware_cheb_normalized(m,std::move(mapped),gate.coefficients);
 auto result=scale_aware_product(m,{std::move(unit),1.},std::move(value));
 auto mask=slot_mask;
 for(auto& slot:mask)slot*=moai_gelu_core_radius/result.gain;
 auto output=m.pm(std::move(result.value),mask);
 for(size_t i=0;i<mask.size();++i)mask[i]=slot_mask[i]*moai_gelu_bias;
 auto bias=k::CKKSEncode(mask,m.r.ckks,&output);
 m.r.ckks->evaluator.add_plain_inplace(output,bias);
 return output;
}

ArionSoftmaxReference ColumnOperators::arion_softmax_reference(const torch::Tensor& scores,const torch::Tensor& values,
                                                             const torch::Tensor& valid) const {
 TORCH_CHECK(method_=="arion"&&!runtime_.calibration.enabled&&valid.dim()==2&&valid.size(1)==128&&
             valid.size(0)>0&&valid.size(0)<=batch_capacity&&
             scores.sizes()==torch::IntArrayRef({valid.size(0),128,128})&&
             values.sizes()==torch::IntArrayRef({valid.size(0),128,64}),"Invalid ARION attention reference");
 const auto& profile=profiles_.at(layer_);
 auto evaluate=[](const torch::Tensor& x,const ResearchPolynomial& p){
  auto z=(2.*x-p.low-p.high)/(p.high-p.low);
  auto a=torch::zeros_like(z),b=torch::zeros_like(z);
  for(int j=int(p.coefficients.size())-1;j>0;--j){
   auto next=2.*z*a-b+p.coefficients[j];
   b=a;a=std::move(next);
  }
  return z*a-b+p.coefficients[0];
 };
 auto pair=valid.unsqueeze(1)&valid.unsqueeze(2);
 auto keys=valid.sum(1,true).clamp_min(1).unsqueeze(-1);
 auto mean=scores.masked_fill(~pair,0.).sum(-1,true)/keys;
 auto variance=(scores-mean).square().masked_fill(~pair,0.).sum(-1,true)/keys;
 auto centered=scores-mean-profile.statmax*(1.25+.1*variance);
 auto active=centered.masked_fill(~pair,0.);
 const auto& basis=arion_exp_working_basis(profile,true);
 TORCH_CHECK(active.min().item<double>()>=basis.low&&active.max().item<double>()<=basis.high,
             "ARION batch exponential inputs outside the public working basis");
 auto e=evaluate(active,basis).masked_fill(~pair,0.);
 auto sum=e.sum(-1,true).masked_fill(~valid.unsqueeze(-1),1.);
 auto inverse=evaluate(sum,arion_poly("inverse",profile.inverse_low,profile.inverse_high,255));
 for(int i=0;i<profile.inverse_iterations;++i)inverse=inverse*(2.-sum*inverse);
 auto av=torch::matmul(e,values);
 return {e,av,av*inverse};
}

CipherList ColumnOperators::batch_arion_softmax(CipherList scores,CipherList av,
                                               const torch::Tensor& valid,int head){
 TORCH_CHECK(method_=="arion"&&!runtime_.calibration.enabled&&scores.size()==128&&av.size()==64&&
             valid.dim()==2&&valid.size(1)==128&&valid.size(0)>0&&valid.size(0)<=batch_capacity&&
             head>=0&&head<12,"Invalid ARION attention batch");
 const auto& profile=profiles_.at(layer_);
 auto mask=batch_token_mask(valid);
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 // Four-level QK outputs reach a bootstrap during the masked variance calculation.
 for(auto& column:scores)if(column.coeff_modulus_size()<=5)column_bounded_refresh(m,column,arion_batch_score_bound);
 auto e=column_exponentials(m,std::move(scores),{},0,true,layer_,profile,&mask);
 auto inverse=column_reciprocal(m,column_denominator(m,e,true,{},0,&mask),true,profile,
                               arion_refresh_bounds(layer_,head));
 // ARION applies the row reciprocal to the value columns after AV.
 for(auto& column:av)column=column_product(m,std::move(column),inverse);
 e.insert(e.end(),std::make_move_iterator(av.begin()),std::make_move_iterator(av.end()));
 return e;
}

CipherList ColumnOperators::batch_encoder(CipherList hidden,const std::vector<LayerWeights>& weights,
                                          torch::Tensor valid){
 const bool arion=method_=="arion"&&!runtime_.calibration.enabled&&!arion_exp_rebased();
 const bool calibrated=runtime_.calibration.enabled&&(method_=="moai"||method_=="euston")&&runtime_.calibration.method==method_;
 TORCH_CHECK((arion||calibrated)&&hidden.size()==768,"Batch encoder requires native ARION or train-calibrated MOAI/Euston columns");
 TORCH_CHECK(valid.dim()==2&&valid.size(0)>0&&valid.size(0)<=batch_capacity&&valid.size(1)==128&&
             !weights.empty()&&weights.size()<=profiles_.size(),"Invalid ARION batch geometry");
 auto active=batch_token_mask(valid);
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=batch_capacity;
 for(size_t i=0;i<weights.size();++i){
  const auto* layer=calibrated?&runtime_.calibration.layers[i]:nullptr;
  hidden=native_column_block(m,std::move(hidden),weights[i],{},arion,int(i),profiles_[i],&active,layer);
  if(runtime_.layer_completed)runtime_.layer_completed(i+1,weights.size());
 }
 return hidden;
}

void register_column_methods(const std::string& profile){for(auto name:{"moai","arion"})MethodRegistry::instance().add({name,"PhantomBootstrapping_common",std::string(name)=="arion"?"arion_column_rotated_diagonals_double_bsgs12_deferred_normalization":"moai_column_diagonal_ccmm_bsgs12",std::string(name)=="arion"?"bd5fcd7f29af2f35c954dc20c83accdc6b8e2a14":"MOAI_GPU_Release-77B9",{}},[name,profile](Runtime& r){return std::make_unique<ColumnOperators>(r,name,std::string(name)=="arion"?profile:"");});}
}
