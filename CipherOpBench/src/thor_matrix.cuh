#pragma once
#include "literature_math.cuh"
namespace cipheropbench::detail {
inline Cipher thor_imaginary(Math& m,Cipher x,double sign=1.) {
  PhantomPlaintext unit;
  m.r.ckks->encoder.encode(k::FlatVec(slots,make_cuDoubleComplex(0.,sign)),x.chain_index(),1.,unit);
  m.r.ckks->evaluator.multiply_plain_inplace(x,unit);return x;
}
inline Cipher thor_conjugate(Math& m,Cipher x) {
  Cipher result;m.r.ckks->evaluator.complex_conjugate(x,*m.r.ckks->galois_keys,result);return result;
}
// Move the head bit through two coordinate bits per stage; inverse restores the public half packing.
inline Cipher thor_head_interleave(Math& m,Cipher x,bool inverse=false) {
  for(int stage=0;stage<7;++stage){int bit=inverse?2+2*stage:14-2*stage;
    x=m.route({x},[=](int i){int shift=bit-2,v=(i>>shift)&7;
      int permuted=inverse?((v<<1)&7)|(v>>2):(v>>1)|((v&1)<<2);
      return std::tuple{0,(i&~(7<<shift))|(permuted<<shift),1.};});
  }return x;
}
// THOR Algorithm 2, specialized to n=128,H=2,c=128: only the column-wrap bucket needs a giant rotation.
inline Cipher thor_complex_ccmm(Math& m,Cipher a,Cipher b) {
  a=thor_head_interleave(m,std::move(a));b=thor_head_interleave(m,std::move(b));
  // Refresh real operands before complexification; the common bootstrap has a real-valued contract.
  // Reserve levels through inverse interleave; avoid refreshing the small AV output.
  m.need(a,10);m.need(b,10);
  auto paired=m.route({a},[](int i){int h=i%2,t=(i/2)%128,d=i/256;
    return std::tuple{0,((d+64)%128)*256+((t+64)%128)*2+h,1.};});
  auto ac=m.scale(m.add(a,thor_imaginary(m,std::move(paired))),.5);
  auto bc=m.add(b,thor_imaginary(m,m.rot(b,16384),-1.));
  Vec row_mask(slots,0.);std::fill_n(row_mask.begin(),256,1.);
  Cipher buckets[2];bool initialized[2]{false,false};
  for(int ell=0;ell<64;++ell){
    auto copied=m.pm(m.rot(bc,ell*256),row_mask);
    for(int step=256;step<slots;step*=2)copied=m.add(copied,m.rot(copied,-step));
    auto shifted=m.rot(ac,(-128*ell+ell)*2);m.align(shifted,copied);Cipher product;
    m.r.record("thor_ccmm_tensor_product",[&]{m.r.ckks->evaluator.multiply(shifted,copied,product);});
    for(int branch=0;branch<2;++branch){
      Vec mask(slots);for(int i=0;i<slots;++i)mask[i]=(((i/2)%128)<128-ell)==(branch==0)?1.:0.;
      PhantomPlaintext plain;
      if(m.prime_scale){const auto& moduli=m.r.ckks->context->get_context_data(product.chain_index()).parms().coeff_modulus();double scale=double(moduli.back().value())*double(moduli[moduli.size()-2].value())*k::scale/product.scale();m.r.ckks->encoder.encode(mask,product.chain_index(),scale,plain);}
      else plain=k::CKKSEncode(mask,m.r.ckks,&product,true);Cipher term;
      m.r.ckks->evaluator.multiply_plain(product,plain,term);
      if(!initialized[branch]){buckets[branch]=std::move(term);initialized[branch]=true;}
      else m.r.ckks->evaluator.add_inplace(buckets[branch],term);
    }
  }
  for(auto& sum:buckets){
    m.r.record("thor_ccmm_lazy_relinearize",[&]{m.r.ckks->evaluator.relinearize_inplace(sum,*m.r.ckks->relin_keys);});
    m.r.ckks->evaluator.rescale_to_next_inplace(sum);m.r.ckks->evaluator.rescale_to_next_inplace(sum);sum.set_scale(k::scale);
  }
  auto result=m.add(buckets[0],m.rot(buckets[1],-256));
  result=m.add(result,thor_conjugate(m,result));
  return thor_head_interleave(m,std::move(result),true);
}
// Pair real input feature blocks as X0+iX1 and W0-iW1, then evaluate all 255 diagonals with BSGS(16).
inline CipherList thor_complex_pcmm(Math& m,CipherList x,const LinearWeights& weights) {
  CipherList packed;Vec first(slots,0.);std::fill_n(first.begin(),half,1.);
  for(auto& ct:x){m.need(ct,2);auto real=m.pm(ct,first),imag=m.pm(m.rot(ct,half),first);
    auto z=m.add(real,thor_imaginary(m,std::move(imag)));z=m.add(z,m.rot(z,half));m.need(z,1);packed.push_back(std::move(z));}
  std::vector<CipherList> baby(3);
  for(int input=0;input<3;++input)for(int step=0;step<16;++step)baby[input].push_back(m.rot(packed[input],step));
  auto weight=weights.weight.contiguous();auto w=weight.accessor<double,2>();CipherList output;
  for(int outpair=0;outpair<3;++outpair){Cipher total;bool any=false;
    for(int giant=-8;giant<8;++giant){Cipher group;bool active=false;
      for(int input=0;input<3;++input)for(int small=0;small<16;++small){int diagonal=16*giant+small;
        if(diagonal < -127 || diagonal >127)continue;k::FlatVec coefficients(slots);bool nonzero=false;
        for(int i=0;i<slots;++i){int source=(i-16*giant+slots)%slots,h=source/half,col=source%128,row=col+diagonal;
          if(row<0||row>=128){coefficients[i]=make_cuDoubleComplex(0.,0.);continue;}
          double real=w[256*input+row][128*(2*outpair+h)+col],imag=-w[256*input+128+row][128*(2*outpair+h)+col];
          coefficients[i]=make_cuDoubleComplex(real*.5,imag*.5);nonzero|=real!=0.||imag!=0.;
        }
        if(!nonzero)continue;auto plain=k::CKKSEncode(coefficients,m.r.ckks,&baby[input][small],true);Cipher term;
        m.r.ckks->evaluator.multiply_plain(baby[input][small],plain,term);
        if(!active){group=std::move(term);active=true;}else m.r.ckks->evaluator.add_inplace(group,term);
      }
      if(!active)continue;m.r.ckks->evaluator.rescale_to_next_inplace(group);group.set_scale(k::scale);
      auto term=m.rot(group,16*giant);total=any?m.add(total,term):std::move(term);any=true;
    }
    if(!any)total=m.zero(packed[0]);total=m.add(total,thor_conjugate(m,total));output.push_back(std::move(total));
  }
  auto bias=k::row_pack_768x1(weights.bias);for(int i=0;i<3;++i){auto plain=k::CKKSEncode(bias[i],m.r.ckks,&output[i]);m.r.ckks->evaluator.add_plain_inplace(output[i],plain);}
  return output;
}
}
