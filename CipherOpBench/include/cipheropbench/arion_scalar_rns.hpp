#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cipheropbench::detail {
// A real constant embeds as the constant polynomial; its NTT is constant per prime.
inline int64_t arion_scalar_coefficient(double value,double scale){
 double product=value*scale;
 if(!std::isfinite(value)||!std::isfinite(scale)||scale<=0||!std::isfinite(product)||std::abs(product)>9007199254740991.)
  throw std::invalid_argument("Scalar RNS requires finite positive scale and abs(value*scale)<=2^53-1");
 return int64_t(std::round(product));
}
struct ArionScalarRNS { uint64_t value[15]{},shoup[15]{}; int levels=0; };
inline ArionScalarRNS arion_scalar_rns(double value,double scale,const std::vector<uint64_t>& moduli){
 if(moduli.empty()||moduli.size()>15)throw std::invalid_argument("Scalar RNS requires 1..15 primes");
 auto coefficient=arion_scalar_coefficient(value,scale);ArionScalarRNS out;out.levels=int(moduli.size());
 for(size_t i=0;i<moduli.size();++i){
  uint64_t q=moduli[i];if(q<2||q>=(uint64_t(1)<<60))throw std::invalid_argument("Scalar RNS modulus must be in [2,2^60)");
  uint64_t r=uint64_t(coefficient<0?-coefficient:coefficient)%q;
  out.value[i]=coefficient<0&&r?q-r:r;out.shoup[i]=uint64_t((__uint128_t(out.value[i])<<64)/q);
 }return out;
}
}
