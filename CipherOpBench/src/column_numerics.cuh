#pragma once
#include "literature_math.cuh"
namespace cipheropbench::detail {
// All common chain primes are close to S or 2048*S. One exact integer multiply
// restores the scale without consuming a second level (public rounding < 2^-18).
inline Cipher column_product(Math& m,Cipher a,Cipher b){
 m.need(a);m.need(b);m.align(a,b);double q=m.r.ckks->context->get_context_data(a.chain_index()).parms().coeff_modulus().back().value();
 Cipher c;m.r.ckks->evaluator.multiply(a,b,c);m.r.ckks->evaluator.relinearize_inplace(c,*m.r.ckks->relin_keys);m.r.ckks->evaluator.rescale_to_next_inplace(c);
 m.integer_scale(c,std::llround(q/k::scale));c.set_scale(k::scale);return c;
}
inline void column_bounded_refresh(Math& m,Cipher& x,double bound){
 // The common bootstrap accepts real bounded values. Affine centering and
 // conjugation remove its imaginary residue before a subsequent square.
 x=m.addc(m.scale(std::move(x),1/(8*bound)),.25);m.r.refresh(x);m.double_real(x);m.integer_scale(x,std::llround(4*bound));x=m.addc(std::move(x),-2*bound);
}
inline Cipher moai_column_inverse(Math& m,Cipher x){
 if(x.coeff_modulus_size()<16)column_bounded_refresh(m,x,2.);m.r.ckks->evaluator.negate_inplace(x);auto a=m.addc(std::move(x),1.),y=m.addc(a,1.);
 for(int i=0;i<16;++i){
  // This is the same released 16-step product. At step 5, |a|<=1 and
  // 0<=y<=64 for a denominator in (0,2), before the reciprocal grows large.
  if(i==5){column_bounded_refresh(m,a,1.);column_bounded_refresh(m,y,64.);}
  a=column_product(m,a,a);y=column_product(m,y,m.addc(a,1.));
 }return y;
}
}
