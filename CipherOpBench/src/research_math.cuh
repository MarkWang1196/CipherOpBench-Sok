#pragma once
#include "extended_matrix.cuh"
#include "thor_matrix.cuh"
#include "cipheropbench/research.cuh"
#include <numbers>
namespace cipheropbench::detail {
inline Vec research_coefficients(const std::function<double(double)>& fn,double low,double high,int degree){
 Vec c(degree+1);for(int j=0;j<=degree;++j){double t=std::numbers::pi*(j+.5)/(degree+1),v=fn((low+high)/2+(high-low)/2*cos(t));for(int k=0;k<=degree;++k)c[k]+=v*cos(k*t)*2/(degree+1);}c[0]*=.5;return c;
}
inline Cipher research_cheb(Math& m,Cipher x,const ResearchPolynomial& p){
 TORCH_CHECK(p.low<p.high&&p.coefficients.size()>=2&&p.coefficients.size()<=1024,"Invalid research polynomial");
 x=m.addc(m.scale(x,2/(p.high-p.low)),-(p.low+p.high)/(p.high-p.low));
 int depth=int(ceil(log2(p.coefficients.size())))+3;
 if(x.coeff_modulus_size()<=size_t(depth)){x=m.scale(m.addc(x,2.),.125);m.r.refresh(x);m.double_real(x);x=m.addc(m.scale(x,4.),-2.);}
 // Chebyshev Paterson-Stockmeyer: dyadic quotient/remainder blocks.
 // P=Q*T_split+R uses T_a*T_b=(T_(a+b)+T_|a-b|)/2.
 // Small baby bases and logarithmic giant bases avoid retaining degree-many ciphertexts.
 constexpr int baby=16;std::map<int,Cipher> basis;basis[0]=m.addc(m.zero(x),1.);basis[1]=x;
 std::function<Cipher(int)> term=[&](int j)->Cipher{auto found=basis.find(j);if(found!=basis.end())return found->second;
  auto y=j%2?m.mul(term(j/2),term(j/2+1)):m.square(term(j/2));auto z=j%2?m.sub(m.add(y,y),x):m.addc(m.add(y,y),-1.);basis.emplace(j,z);return z;};
 std::function<Cipher(Vec)> evaluate=[&](Vec c)->Cipher{
  while(c.size()>1&&c.back()==0.)c.pop_back();
  if(c.size()<=baby){auto result=m.addc(m.zero(x),c[0]);for(size_t j=1;j<c.size();++j)if(c[j]!=0.)result=m.add(result,m.scale(term(int(j)),c[j]));return result;}
  int degree=int(c.size())-1,split=1;while(2*split<=degree)split*=2;Vec q(degree-split+1);
  for(int j=degree;j>=split;--j){double lead=c[j];if(lead==0.)continue;c[j]=0.;int d=j-split;if(d==0)q[0]+=lead;else{q[d]+=2*lead;c[std::abs(2*split-j)]-=lead;}}
  c.resize(split);auto left=evaluate(q),right=evaluate(c);return m.add(m.mul(left,term(split)),right);
 };return evaluate(p.coefficients);
}
inline ResearchPolynomial research_polynomial(const std::function<double(double)>& f,double low,double high,int d){return {low,high,research_coefficients(f,low,high,d)};}
inline Cipher head_row_rotation(Math& m,const Cipher& x,int shift){shift=(shift%128+128)%128;if(!shift)return x;
 return m.route({x},[=](int i){int h=i/half,t=(i%half)/128,c=i%128;return std::tuple{0,h*half+((t+shift)%128)*128+c,1.};});
}
inline Cipher real_part(Math& m,Cipher x){return m.scale(m.add(x,thor_conjugate(m,x)),.5);}
inline Cipher imaginary_part(Math& m,Cipher x){return m.scale(thor_imaginary(m,m.sub(x,thor_conjugate(m,x)),-1.),.5);}
inline CipherList spatial_inputs(Math& m,CipherList x){for(auto& c:x){m.need(c,12);c=tile_transpose(m,c);}return x;}
inline Cipher column(Math& m,const CipherList& spatial,int feature){
 Vec mask(slots);std::fill_n(mask.begin(),128,1.);auto y=m.pm(m.rot(spatial.at(feature/256),(feature%256)*128),mask);
 for(int step=128;step<slots;step*=2)y=m.add(y,m.rot(y,-step));return y;
}
inline void add_bias(Math& m,CipherList& x,torch::Tensor bias){auto packed=k::row_pack_768x1(bias);for(int g=0;g<3;++g){auto p=k::CKKSEncode(packed[g],m.r.ckks,&x[g]);m.r.ckks->evaluator.add_plain_inplace(x[g],p);}}
}
