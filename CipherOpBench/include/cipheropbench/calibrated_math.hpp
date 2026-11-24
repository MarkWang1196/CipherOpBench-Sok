#pragma once
#include "calibration.hpp"
#include "literature_coefficients.cuh"
namespace cipheropbench::calibration {
using Vec=std::vector<double>;
inline Vec power_to_series(const Vec& coefficients){
 Vec series(coefficients.size(),0.),power(coefficients.size(),0.);power[0]=1.;
 for(size_t j=0;j<coefficients.size();++j){
  for(size_t k=0;k<=j;++k)series[k]+=coefficients[j]*power[k];
  Vec next(power.size(),0.);for(size_t k=0;k<=j&&k+1<power.size();++k){if(k==0)next[1]+=power[0];else{next[k-1]+=power[k]/2;next[k+1]+=power[k]/2;}}power=std::move(next);
 }return series;
}
template<class M,class C> C series(M& m,C x,const Vec& coefficients){
 m.need(x,int(std::ceil(std::log2(coefficients.size())))+2);
 std::vector<C> basis(coefficients.size());basis[0]=m.addc(m.zero(x),1.);basis[1]=x;
 for(size_t j=2;j<basis.size();++j){auto v=j%2?m.mul(basis[j/2],basis[j/2+1]):m.square(basis[j/2]);basis[j]=j%2?m.sub(m.add(v,v),x):m.addc(m.add(v,v),-1.);}
 auto result=m.addc(m.zero(x),coefficients[0]);
 for(size_t j=1;j<coefficients.size();++j)if(std::abs(coefficients[j])>1e-14)result=m.add(result,m.scale(basis[j],coefficients[j]));
 return result;
}
template<class M,class C> C sign(M& m,C x){
 const Vec g{0,5850./1024,0,-34974./1024,0,97015./1024,0,-113492./1024,0,46623./1024};
 const Vec f{0,315./128,0,-420./128,0,378./128,0,-180./128,0,35./128};
 for(int i=0;i<4;++i)x=m.poly(x,g);for(int i=0;i<3;++i)x=m.poly(x,f);return x;
}
template<class M,class C> C exp_euston(M& m,C x,int squares){auto y=m.poly(m.scale(x,std::exp2(-squares)),literature::EUSTON_EXP);for(int i=0;i<squares;++i)y=m.square(y);return y;}
template<class M,class C> C gelu(M& m,C value,const Layer& p,const std::string& method){
 auto below=sign(m,m.scale(m.addc(value,8.),1/(p.gelu_bound+8.)));
 auto above=sign(m,m.scale(m.addc(value,-8.),1/(p.gelu_bound+8.)));
 auto inside=m.scale(m.sub(below,above),.5);
 auto x=m.add(m.mul(value,inside),m.scale(m.add(below,above),4.));C y;
 if(method=="nexus"||method=="moai")y=series(m,m.scale(x,1/8.5),p.gelu_coefficients);
 else if(method=="thor"){
  auto first=series(m,m.scale(x,1./64),power_to_series(literature::THOR_GELU_P1));
  y=m.mul(x,m.addc(m.scale(series(m,first,power_to_series(literature::THOR_GELU_P2)),.5),.5));
 }else{
  const Vec g{0,5850./1024,0,-34974./1024,0,97015./1024,0,-113492./1024,0,46623./1024};
  const Vec f{0,315./128,0,-420./128,0,378./128,0,-180./128,0,35./128};
  auto s=m.scale(x,1./9);for(int i=0;i<3;++i)s=m.poly(s,g);for(int i=0;i<3;++i)s=m.poly(s,f);
  auto a=m.mul(x,s),den=m.addc(exp_euston(m,m.scale(a,-1.702),p.euston_gelu_squarings),1.);
  auto inv=m.inverse(den,1.,3);
  y=m.mul(m.mul(m.mul(a,inv),s),exp_euston(m,m.scale(m.mul(m.addc(s,-1.),a),.851),p.euston_gelu_squarings));
 }
 return m.add(m.mul(y,inside),m.mul(value,m.scale(m.addc(above,1.),.5)));
}
// Reciprocal is physically divided by 128 so refresh never sees a huge inverse.
template<class M,class C> C scaled_reciprocal(M& m,C denominator,double maximum){
 auto error=m.addc(m.scale(denominator,-1./maximum),1.);
 auto result=m.scale(m.addc(error,1.),1./128);
 for(int i=0;i<14;++i){error=m.square(error);result=m.mul(result,m.addc(error,1.));}
 return result;
}
template<class M,class C> C softmax_base(M& m,C scores,const Layer& p,const std::string& method){
 int count=method=="euston"?p.euston_softmax_squarings:p.exp_squarings;
 auto z=m.scale(m.addc(scores,-p.softmax_shift),std::exp2(-count));
 return method=="euston"?m.poly(z,literature::EUSTON_EXP):m.addc(z,1.);
}
template<class M,class C> C asor_isqrt(M& m,C x,double maximum,double epsilon){
 double delta=std::sqrt(epsilon);auto a=m.scale(x,1/maximum),b=m.addc(m.zero(a),delta);
 for(int i=0;epsilon<.999;++i){
  if(i>=80)throw std::runtime_error("Calibrated inverse sqrt did not converge");
  m.need(a,5);m.need(b,5);
  double aa=1-std::pow(epsilon,3),bb=6*epsilon*epsilon-6,cc=9-9*epsilon;
  double k=2*cc/(-bb+std::sqrt(std::max(bb*bb-4*aa*cc,0.)));
  auto correction=m.addc(m.scale(a,-1.),3/k);
  b=m.scale(m.mul(b,correction),std::pow(k,1.5)/2);
  a=m.scale(m.mul(a,m.square(correction)),std::pow(k,3)/4);
  epsilon=k*epsilon*std::pow(3-k*epsilon,2)/4;
 }return m.scale(b,1/(delta*std::sqrt(maximum)));
}
template<class M,class C> C inverse_sqrt(M& m,C variance,const Layer& p,int kind,const std::string& method){
 double maximum=p.high.at(kind-1);if(method=="thor")return asor_isqrt(m,variance,maximum,p.low.at(kind-1)/maximum);
 auto u=m.scale(variance,100/maximum);C estimate;
 if(method=="euston"){
  auto d=m.addc(m.scale(u,.01),-1.);
  auto root=m.add(m.addc(m.scale(d,.5),1.),m.scale(m.square(d),-.125));
  estimate=m.scale(m.inverse(root,2.,4),.1);
 }else estimate=m.addc(m.scale(u,-.000129054537),.129054537);
 for(int i=0;i<p.iterations.at(kind-1);++i){
  m.need(estimate,5);
  auto correction=m.addc(m.scale(m.mul(u,m.square(estimate)),-.5),1.5);
  estimate=m.mul(estimate,correction);
 }return m.scale(estimate,std::sqrt(100/maximum));
}
}
