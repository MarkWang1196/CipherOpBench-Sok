#pragma once
#include "research_math.cuh"
#include "column_numerics.cuh"
namespace cipheropbench::detail {
inline Cipher arion_inverse_polynomial(Math& m,Cipher x){
 auto p=research_polynomial([](double z){return 1/z;},.03,300.,255);x=m.addc(m.scale(x,2/(p.high-p.low)),-(p.low+p.high)/(p.high-p.low));
 std::map<int,Cipher> basis;basis[1]=x;
 std::function<Cipher(int)> term=[&](int j)->Cipher{auto found=basis.find(j);if(found!=basis.end())return found->second;auto product=column_product(m,term(j/2),term(j%2?j/2+1:j/2));auto y=j%2?m.sub(m.add(product,product),x):m.addc(m.add(product,product),-1.);basis.emplace(j,y);return y;};
 // Dyadic Chebyshev quotient/remainder keeps baby16 and giant powers, rather than255 ciphertexts.
 std::function<Cipher(std::vector<double>)> evaluate=[&](std::vector<double> c)->Cipher{
  while(c.size()>1&&c.back()==0.)c.pop_back();
  if(c.size()<=16){auto y=m.addc(m.zero(x),c[0]);for(size_t j=1;j<c.size();++j)if(c[j]!=0.)y=m.add(y,m.scale(term(int(j)),c[j]));return y;}
  int degree=int(c.size())-1,split=1;while(2*split<=degree)split*=2;std::vector<double> q(degree-split+1);
  for(int j=degree;j>=split;--j){double lead=c[j];if(lead==0.)continue;c[j]=0.;int d=j-split;if(d==0)q[0]+=lead;else{q[d]+=2*lead;c[std::abs(2*split-j)]-=lead;}}
  c.resize(split);auto left=evaluate(q),right=evaluate(c);return m.add(column_product(m,std::move(left),term(split)),std::move(right));
 };return evaluate(p.coefficients);
}
}
