#pragma once
#include "cipheropbench/literature.cuh"
#include "cipheropbench/literature_coefficients.cuh"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <tuple>
namespace cipheropbench::detail {
using Vec=std::vector<double>;
constexpr int slots=32768,half=16384;
struct Math {
  Runtime& r;
  std::function<void(int,const Cipher&,const Cipher&,double,double)> inverse_observer;
  // Match the dropped prime when normalizing scale; avoid accumulating S/q gain.
  bool prime_scale = false;
  int column_token_stride = 1;
  int level_floor()const{const char* guard=std::getenv("CIPHEROPBENCH_LEGACY_GUARD_LOW_LEVELS");return guard&&std::string(guard)=="1"?3:0;}
  void need(Cipher& x,int depth=1){if(x.coeff_modulus_size()<=size_t(depth+level_floor()))r.refresh(x);}
  Cipher rot(Cipher x,int shift){int s=(shift%slots+slots)%slots;if(!s)return x;Cipher y;r.ckks->evaluator.rotate_vector(x,s,*r.ckks->galois_keys,y);return y;}
  void align(Cipher& a,Cipher& b){k::mod_switch_to_same(a,b,r.ckks);a.set_scale(k::scale);b.set_scale(k::scale);}
  Cipher add(Cipher a,Cipher b){align(a,b);r.ckks->evaluator.add_inplace(a,b);return a;}
  Cipher sub(Cipher a,Cipher b){align(a,b);r.ckks->evaluator.sub_inplace(a,b);return a;}
  Cipher addc(Cipher a,double b){auto p=k::CKKSEncode(Vec(slots,b),r.ckks,&a);r.ckks->evaluator.add_plain_inplace(a,p);return a;}
  Cipher pm(Cipher a,const Vec& v){need(a);PhantomPlaintext p;if(prime_scale){double q=r.ckks->context->get_context_data(a.chain_index()).parms().coeff_modulus().back().value();r.ckks->encoder.encode(v,a.chain_index(),q*k::scale/a.scale(),p);}else p=k::CKKSEncode(v,r.ckks,&a,true);r.ckks->evaluator.multiply_plain_inplace(a,p);r.ckks->evaluator.rescale_to_next_inplace(a);a.set_scale(k::scale);return a;}
  Cipher scale(Cipher a,double c){return pm(std::move(a),Vec(slots,c));}
  Cipher mul(Cipher a,Cipher b){need(a,prime_scale?2:1);need(b,prime_scale?2:1);align(a,b);Cipher c;r.ckks->evaluator.multiply(a,b,c);r.ckks->evaluator.relinearize_inplace(c,*r.ckks->relin_keys);r.ckks->evaluator.rescale_to_next_inplace(c);if(prime_scale)return pm(std::move(c),Vec(slots,1.));c.set_scale(k::scale);return c;}
  Cipher square(Cipher x){need(x,prime_scale?2:1);return mul(x,x);}
  Cipher zero(Cipher x){return scale(std::move(x),0.);}
  Cipher poly(Cipher x,const Vec& c){
    TORCH_CHECK(c.size()>=2&&c.size()<=32,"Polynomial degree outside 1..31");
    need(x,6);auto x2=square(x),x4=square(x2);Cipher x8,x16;if(c.size()>8)x8=square(x4);if(c.size()>16)x16=square(x8);
    std::vector<Cipher> parts;
    for(size_t j=0;j<c.size();j+=4){auto y=addc(zero(x),c[j]);if(j+1<c.size())y=add(y,scale(x,c[j+1]));if(j+2<c.size())y=add(y,scale(x2,c[j+2]));if(j+3<c.size())y=add(y,mul(x2,scale(x,c[j+3])));parts.push_back(std::move(y));}
    for(const auto* power:{&x4,&x8,&x16}){if(parts.size()==1)break;std::vector<Cipher> next;for(size_t j=0;j<parts.size();j+=2)next.push_back(j+1<parts.size()?add(parts[j],mul(parts[j+1],*power)):parts[j]);parts=std::move(next);}return parts[0];
  }
  Cipher chebyshev(Cipher x,const Vec& coefficients){
    // The same released polynomial in a bounded basis avoids cancellation at the common 40-bit scale.
    Vec series(coefficients.size(),0.),power(coefficients.size(),0.);power[0]=1.;
    for(size_t j=0;j<coefficients.size();++j){
      for(size_t k=0;k<=j;++k)series[k]+=coefficients[j]*power[k];
      Vec next(power.size(),0.);for(size_t k=0;k<=j&&k+1<power.size();++k){if(k==0)next[1]+=power[0];else{next[k-1]+=power[k]/2;next[k+1]+=power[k]/2;}}power=std::move(next);
    }
    need(x,7);CipherList basis(coefficients.size());basis[0]=addc(zero(x),1.);basis[1]=x;
    for(size_t j=2;j<basis.size();++j){
      auto product=j%2?mul(basis[j/2],basis[j/2+1]):square(basis[j/2]);
      basis[j]=j%2?sub(add(product,product),x):addc(add(product,product),-1.);
    }
    auto result=addc(zero(x),series[0]);
    for(size_t j=1;j<series.size();++j)if(series[j]!=0.)result=add(result,scale(basis[j],series[j]));
    return result;
  }

  Cipher inverse(Cipher x,double divisor,int iterations){auto a=addc(scale(x,-1./divisor),1.);auto y=addc(a,1.);for(int i=0;i<iterations;++i){a=square(a);y=mul(y,addc(a,1.));}return scale(y,1./divisor);}
  Cipher exp_euston(Cipher x,int squarings=1){auto y=poly(scale(x,std::ldexp(1.,-squarings)),literature::EUSTON_EXP);for(int i=0;i<squarings;++i)y=square(y);return y;}
  void integer_scale(Cipher& x,int64_t factor){
    if(factor==1)return;PhantomPlaintext p;
    r.ckks->encoder.encode(Vec(slots,double(factor)),x.chain_index(),1.,p);
    r.ckks->evaluator.multiply_plain_inplace(x,p);
  }
  void double_real(Cipher& x){Cipher conjugate;r.ckks->evaluator.complex_conjugate(x,*r.ckks->galois_keys,conjugate);r.ckks->evaluator.add_inplace(x,conjugate);}
  std::pair<Cipher,double> adaptive_inverse(Cipher x,double maximum,double epsilon,double alpha=0.001){
    auto b=maximum==1.?x:scale(x,1./maximum),a=addc(sub(b,b),1.);
    double delta_a=1.,delta_b=1.,upper_a=1.,initial_epsilon=epsilon;
    auto real_pair=[&]{double_real(a);double_real(b);delta_a*=2;delta_b*=2;upper_a*=2;};
    for(int i=0;epsilon<1-alpha;++i){
      TORCH_CHECK(i<80,"aSOR reciprocal did not converge");double k=2/(1+epsilon),next_epsilon=k*epsilon*(2-k*epsilon);
      if(std::min(a.coeff_modulus_size(),b.coeff_modulus_size())<=size_t(2+level_floor())){
        real_pair();double factor=std::exp2(-std::ceil(std::log2(std::max(upper_a,1.))));
        if(factor<1){a=scale(a,factor);delta_a*=factor;upper_a*=factor;}
        auto adjust=std::max<int64_t>(1,int64_t(1/(delta_b*8)));integer_scale(b,adjust);delta_b*=adjust;
        r.refresh(a);r.refresh(b);upper_a+=0.001;real_pair();
      }
      auto correction=b;r.ckks->evaluator.negate_inplace(correction);correction=addc(correction,2/k*delta_b);
      a=mul(a,correction);b=mul(b,correction);upper_a*=2/k*delta_b;
      delta_a=delta_a*delta_b/(k*k);delta_b=delta_b*delta_b/(k*k);epsilon=next_epsilon;
      // A larger physical delta avoids amplifying rescale noise at the shared 40-bit scale.
      auto adjust=int64_t(1/(delta_b*4));if(adjust>1){real_pair();adjust=std::max<int64_t>(1,int64_t(1/(delta_b*4)));}else adjust=1;
      integer_scale(a,adjust);integer_scale(b,adjust);delta_a*=adjust;delta_b*=adjust;upper_a*=adjust;
      upper_a=std::min(upper_a,1.01*delta_a/initial_epsilon);
      if(inverse_observer)inverse_observer(i,a,b,delta_a,delta_b);
    }
    // Native-style delta bookkeeping keeps the numerator away from the CKKS noise floor.
    // Balance the final multiply: avoid amplifying the probability operand before CMUL.
    auto final_adjust=std::max<int64_t>(1,int64_t(1/std::max(upper_a,1e-12)));
    integer_scale(a,final_adjust);delta_a*=final_adjust;
    return {a,1/(delta_a*maximum)};
  }
  Cipher adaptive_isqrt(Cipher x,double maximum,double epsilon){
    TORCH_CHECK(maximum>0&&epsilon>0&&epsilon<1,"Invalid inverse sqrt interval");
    // Keep the physical inverse <= 1 throughout aSOR, including before bootstraps.
    double delta=std::sqrt(epsilon);
    auto a=scale(x,1./maximum),b=addc(zero(a),delta);
    for(int i=0;epsilon<0.999;++i){
      TORCH_CHECK(i<80,"aSOR inverse sqrt did not converge");
      if(std::min(a.coeff_modulus_size(),b.coeff_modulus_size())<=5){r.refresh(a);r.refresh(b);}
      double aa=1-pow(epsilon,3),bb=6*epsilon*epsilon-6,cc=9-9*epsilon;
      double k=2*cc/(-bb+sqrt(std::max(bb*bb-4*aa*cc,0.)));
      auto correction=addc(scale(a,-1),3/k);
      b=scale(mul(b,correction),pow(k,1.5)/2);
      a=scale(mul(a,square(correction)),pow(k,3)/4);
      epsilon=k*epsilon*pow(3-k*epsilon,2)/4;
    }return scale(b,1/(delta*sqrt(maximum)));
  }
  // Sparse cyclic-diagonal transform; all input permutations remain encrypted.
  Cipher route(const CipherList& input,const std::function<std::tuple<int,int,double>(int)>& mapping){
    CipherList prepared=input;for(auto& ct:prepared)need(ct);
    std::map<std::pair<int,int>,std::vector<std::pair<int,double>>> groups;
    for(int i=0;i<slots;++i){auto [source,index,value]=mapping(i);if(source>=0&&value!=0)groups[{source,(index-i+slots)%slots}].emplace_back(i,value);}
    Cipher out;bool first=true;
    for(auto& [key,positions]:groups){Vec mask(slots,0.);for(auto [i,value]:positions)mask[i]=value;auto term=pm(rot(prepared.at(key.first),key.second),mask);out=first?std::move(term):add(out,term);first=false;}
    return first?zero(input.at(0)):out;
  }
  Cipher transpose(const Cipher& x){return route({x},[](int i){int h=i/half,a=i%half;return std::tuple{0,h*half+(a%128)*128+a/128,1.};});}
  // Factor transpose from shear: composing them directly creates O(n^2) rotation diagonals.
  Cipher lower(const Cipher& x,bool transposed){
    auto input=transposed?transpose(x):x;
    return route({input},[](int i){int h=i/half,d=(i%half)/128,t=i%128;return std::tuple{0,h*half+((d+t)%128)*128+t,1.};});
  }
  Cipher unlower(const Cipher& x,bool transposed){
    auto output=route({x},[](int i){int h=i/half,row=(i%half)/128,col=i%128;return std::tuple{0,h*half+((row-col+128)%128)*128+col,1.};});
    return transposed?transpose(output):output;
  }
  Cipher row_diagonal(const Cipher& x,bool inverse,double factor=1.){
    return route({x},[=](int i){int h=i/half,row=(i%half)/128,col=i%128;return std::tuple{0,h*half+row*128+(col+(inverse?128-row:row))%128,factor};});
  }
  CipherList columns(const CipherList& row,const std::vector<int>& indices,int keep_levels=0){
    CipherList transposed;for(auto x:row){
      // Drop unused limbs before creating hundreds of replicated columns.
      if(keep_levels&&x.coeff_modulus_size()>size_t(keep_levels+2))r.ckks->evaluator.mod_switch_to_inplace(x,k::chain_idx(keep_levels+2));
      transposed.push_back(transpose(x));}CipherList out;
    Vec mask(slots,0.);std::fill_n(mask.begin(),128,1.);
    for(int c:indices){int group=c/256,local=c%256;auto y=pm(rot(transposed.at(group),local*128),mask);for(int step=128;step<slots;step*=2)y=add(y,rot(y,-step));if(keep_levels&&y.coeff_modulus_size()>size_t(keep_levels))r.ckks->evaluator.mod_switch_to_inplace(y,k::chain_idx(keep_levels));out.push_back(std::move(y));}return out;
  }
  CipherList pack_columns(const CipherList& cols,int count,double factor=1.){
    TORCH_CHECK(cols.size()==size_t(count)&&count%256==0,"Column packing shape mismatch");CipherList out;
    for(int group=0;group<count/256;++group){Cipher combined;for(int c=0;c<256;++c){Vec mask(slots,0.);std::fill_n(mask.begin()+c*128,128,factor);auto term=pm(cols[group*256+c],mask);combined=c?add(combined,term):std::move(term);}out.push_back(transpose(combined));}return out;
  }
  CipherList cpmm(const CipherList& cols,const LinearWeights& w){
    auto weight=w.weight.contiguous();auto values=weight.accessor<double,2>();CipherList out;
    // Each input column already repeats across all slots: evaluate 256 output columns in parallel.
    for(int group=0;group<weight.size(1)/256;++group){Cipher sum;bool first=true;
      for(int input=0;input<weight.size(0);++input){Vec coefficients(slots);bool nonzero=false;
        for(int column=0;column<256;++column){double value=values[input][group*256+column];nonzero|=value!=0.;std::fill_n(coefficients.begin()+column*128,128,value);}
        if(!nonzero)continue;auto ct=cols.at(input);need(ct);auto pt=k::CKKSEncode(coefficients,r.ckks,&ct,true);Cipher term;
        r.ckks->evaluator.multiply_plain(ct,pt,term);if(first){sum=std::move(term);first=false;}else r.ckks->evaluator.add_inplace(sum,term);
      }
      if(first)sum=zero(cols[0]);else{r.ckks->evaluator.rescale_to_next_inplace(sum);sum.set_scale(k::scale);}
      Vec bias(slots);for(int column=0;column<256;++column)std::fill_n(bias.begin()+column*128,128,w.bias[group*256+column].item<double>());
      auto pt=k::CKKSEncode(bias,r.ckks,&sum);r.ckks->evaluator.add_plain_inplace(sum,pt);out.push_back(transpose(sum));
    }return out;
  }
  Cipher row_sum(Cipher x,double factor=1.){need(x,2);if(!prime_scale&&factor==1.)return k::quick_sum(x,r.ckks,128);
    for(int step=1;step<128;step*=2)x=add(x,rot(x,step));
    Vec mask(slots);for(int i=0;i<slots;i+=128)mask[i]=factor;x=pm(x,mask);
    for(int step=1;step<128;step*=2)x=add(x,rot(x,-step));return x;
  }
  Cipher hidden_sum(const CipherList& x){auto sum=row_sum(x[0]);for(size_t i=1;i<x.size();++i)sum=add(sum,row_sum(x[i]));return add(sum,rot(sum,half));}
};
}
