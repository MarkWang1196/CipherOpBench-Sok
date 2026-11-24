#pragma once
#include "research_math.cuh"
namespace cipheropbench::detail {
struct ScaledPolynomialCipher {Cipher value;double gain=1.;};
inline ScaledPolynomialCipher scale_aware_product(Math& m,ScaledPolynomialCipher a,ScaledPolynomialCipher b){
 m.need(a.value);m.need(b.value);size_t chain=std::max(a.value.chain_index(),b.value.chain_index());
 double q=m.r.ckks->context->get_context_data(chain).parms().coeff_modulus().back().value();
 bool saved=m.prime_scale;m.prime_scale=false;auto z=m.mul(a.value,b.value);m.prime_scale=saved;
 return {z,a.gain*b.gain*k::scale/q};
}
// Public modulus-dependent gains are folded into coefficient encodings, not extra rescalings.
inline ScaledPolynomialCipher scale_aware_cheb_normalized(Math& m,Cipher x,const Vec& coefficients){
 TORCH_CHECK(m.prime_scale&&coefficients.size()>=2&&coefficients.size()<=1024,"Invalid normalized polynomial");
 std::map<int,ScaledPolynomialCipher> basis;basis[0]={m.addc(m.zero(x),1.),1.};basis[1]={x,1.};
 std::function<ScaledPolynomialCipher(int)> term=[&](int j){auto found=basis.find(j);if(found!=basis.end())return found->second;
  auto y=scale_aware_product(m,term(j/2),term(j%2?j/2+1:j/2));
  auto z=j%2?m.sub(m.add(y.value,y.value),m.scale(x,y.gain)):m.addc(m.add(y.value,y.value),-y.gain);
  ScaledPolynomialCipher out{z,y.gain};basis.emplace(j,out);return out;
 };
 std::function<ScaledPolynomialCipher(Vec)> evaluate=[&](Vec c){
  while(c.size()>1&&c.back()==0.)c.pop_back();
  if(c.size()<=16){auto result=m.addc(m.zero(x),c[0]);for(size_t j=1;j<c.size();++j)if(c[j]!=0.){auto b=term(int(j));result=m.add(result,m.scale(b.value,c[j]/b.gain));}return ScaledPolynomialCipher{result,1.};}
  int degree=int(c.size())-1,split=1;while(2*split<=degree)split*=2;Vec quotient(degree-split+1);
  for(int j=degree;j>=split;--j){double lead=c[j];if(lead==0.)continue;c[j]=0.;int d=j-split;if(d==0)quotient[0]+=lead;else{quotient[d]+=2*lead;c[std::abs(2*split-j)]-=lead;}}
  c.resize(split);auto left=evaluate(quotient),right=evaluate(c);auto product=scale_aware_product(m,left,term(split));
  return ScaledPolynomialCipher{m.add(product.value,m.scale(right.value,product.gain/right.gain)),product.gain};
 };return evaluate(coefficients);
}
inline ScaledPolynomialCipher scale_aware_cheb(Math& m,Cipher x,const ResearchPolynomial& p){
 TORCH_CHECK(m.prime_scale&&p.low<p.high&&p.coefficients.size()>=2&&p.coefficients.size()<=1024,"Invalid scale-aware polynomial");
 x=m.addc(m.scale(x,2/(p.high-p.low)),-(p.low+p.high)/(p.high-p.low));
 return scale_aware_cheb_normalized(m,std::move(x),p.coefficients);
}
inline Cipher scale_aware_activation_input(Math& m,Cipher x,const ResearchPolynomial& p){
 TORCH_CHECK(m.prime_scale&&p.low<p.high&&p.coefficients.size()>=2&&p.coefficients.size()<=1024,"Invalid activation polynomial");
 x=m.addc(m.scale(x,2/(p.high-p.low)),-(p.low+p.high)/(p.high-p.low));
 int depth=int(ceil(log2(p.coefficients.size())))+3;
 if(x.coeff_modulus_size()<=size_t(depth)){x=m.scale(m.addc(x,2.),.125);m.r.refresh(x);m.double_real(x);x=m.addc(m.scale(x,4.),-2.);}
 return x;
}
inline Cipher scale_aware_prescaled_activation(Math& m,Cipher x,const ResearchPolynomial& p){
 x=scale_aware_activation_input(m,std::move(x),p);
 // The probe is actual FHE work and remains inside the operator's measured region.
 auto probe=scale_aware_cheb_normalized(m,x,p.coefficients);auto coefficients=p.coefficients;
 for(auto& coefficient:coefficients)coefficient/=probe.gain;
 auto result=scale_aware_cheb_normalized(m,x,coefficients);
 TORCH_CHECK(std::abs(result.gain/probe.gain-1.)<1e-10,"Public rescale schedule changed after coefficient adjustment");
 return result.value;
}
inline CipherList scale_aware_polynomial_norm(Math& m,const CipherList& centered,torch::Tensor weight,torch::Tensor bias){
 TORCH_CHECK(m.prime_scale&&centered.size()==3,"Scale-aware norm requires the common packing");
 CipherList squared,out;double square_gain=0.;
 for(const auto& c:centered){auto square=scale_aware_product(m,{c,1.},{c,1.});if(square_gain)TORCH_CHECK(std::abs(square_gain-square.gain)<1e-12,"Unexpected square scale mismatch");square_gain=square.gain;squared.push_back(square.value);}
 auto variance=m.scale(m.hidden_sum(squared),1./(768*square_gain));auto p=research_polynomial([](double x){return 1/sqrt(x);},.1,10.,119);auto inverse=scale_aware_cheb(m,variance,p);m.need(inverse.value);
 auto packed=k::row_pack_768x1(weight);
 for(int g=0;g<3;++g){auto c=centered[g];m.need(c,2);size_t chain=std::max(c.chain_index()+1,inverse.value.chain_index());double q=m.r.ckks->context->get_context_data(chain).parms().coeff_modulus().back().value();Vec v(slots);for(int i=0;i<slots;++i)v[i]=cuCreal(packed[g][i])*q/(k::scale*inverse.gain);auto weighted=m.pm(c,v);out.push_back(scale_aware_product(m,{weighted,1.},{inverse.value,1.}).value);}
 add_bias(m,out,bias);return out;
}

}
