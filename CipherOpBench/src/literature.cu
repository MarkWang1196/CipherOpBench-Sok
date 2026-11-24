#include "cipheropbench/literature.cuh"
#include "cipheropbench/method_registry.cuh"
#include "cipheropbench/literature_coefficients.cuh"
#include "cipheropbench/nexus_batch_protocol.cuh"
#include "literature_math.cuh"
#include "thor_matrix.cuh"
#include "calibrated_operators.cuh"
#include "column_numerics.cuh"
#include "scale_aware_chebyshev.cuh"
#include <map>
#include <tuple>
#include <numeric>
#include <cmath>
#include <span>
namespace cipheropbench {
namespace {
using Vec=std::vector<double>;
constexpr int slots=32768,half=16384;
constexpr std::array<std::array<int,2>,6> pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
using detail::Math;
const Vec sign_g{0,5850./1024,0,-34974./1024,0,97015./1024,0,-113492./1024,0,46623./1024};
const Vec sign_f{0,315./128,0,-420./128,0,378./128,0,-180./128,0,35./128};

Cipher sign_odd9(Math& m,Cipher x,const Vec& c) {
  if(x.coeff_modulus_size()<=6)detail::column_bounded_refresh(m,x,2.);
  auto prime=[&](const Cipher& value){
    return double(m.r.ckks->context->get_context_data(value.chain_index()).parms().coeff_modulus().back().value());
  };
  auto product=[&](Cipher a,Cipher b){
    k::mod_switch_to_same(a,b,m.r.ckks);Cipher out;
    m.r.ckks->evaluator.multiply(a,b,out);
    m.r.ckks->evaluator.relinearize_inplace(out,*m.r.ckks->relin_keys);
    m.r.ckks->evaluator.rescale_to_next_inplace(out);
    return out;
  };
  auto weighted=[&](Cipher value,double coefficient,double scale){
    PhantomPlaintext pt;
    m.r.ckks->encoder.encode(Vec(slots,coefficient),value.chain_index(),prime(value)*scale/value.scale(),pt);
    m.r.ckks->evaluator.multiply_plain_inplace(value,pt);
    m.r.ckks->evaluator.rescale_to_next_inplace(value);value.set_scale(scale);
    return value;
  };
  auto x2=product(x,x),x3=product(x,x2),x6=product(x3,x3);
  auto middle=weighted(x2,c[5],k::scale*prime(x3)/x3.scale());
  middle=product(m.addc(std::move(middle),c[3]),x3);
  const double high_scale=k::scale*prime(x6)/x6.scale();
  auto high=weighted(x3,c[9],high_scale),linear=weighted(x,c[7],high_scale);
  k::mod_switch_to_same(high,linear,m.r.ckks);
  m.r.ckks->evaluator.add_inplace(high,linear);
  high=product(std::move(high),std::move(x6));
  return m.add(m.add(std::move(high),std::move(middle)),weighted(std::move(x),c[1],k::scale));
}

std::vector<int> range(int count){std::vector<int> x(count);std::iota(x.begin(),x.end(),0);return x;}
CipherList euston_qk_columns(Math& m,std::span<const Cipher> q,std::span<const Cipher> k,int stride){
  TORCH_CHECK(q.size()==64&&k.size()==64,"Euston QK requires 64 query and key columns");
  CipherList result;result.reserve(128);
  for(int d=0;d<128;++d){
    auto sum=m.mul(q[0],m.rot(k[0],d*stride));
    for(int c=1;c<64;++c)sum=m.add(std::move(sum),m.mul(q[c],m.rot(k[c],d*stride)));
    result.push_back(std::move(sum));
  }
  return result;
}
CipherList euston_av_columns(Math& m,std::span<const Cipher> p,std::span<const Cipher> v,int stride){
  TORCH_CHECK(p.size()==128&&v.size()==64,"Euston AV requires 128 diagonals and 64 value columns");
  CipherList result;result.reserve(v.size());
  for(const auto& value:v){
    auto sum=m.mul(p[0],value);
    for(int d=1;d<128;++d)sum=m.add(std::move(sum),m.mul(p[d],m.rot(value,d*stride)));
    result.push_back(std::move(sum));
  }
  return result;
}

std::pair<detail::ScaledPolynomialCipher,double> scaled_goldschmidt(Math& m,Cipher sum,int iterations,
                                                                  bool track_complement=false,
                                                                  double denominator_gain=1.) {
  if(sum.coeff_modulus_size()<=5)detail::column_bounded_refresh(m,sum,2.*denominator_gain);
  if(!track_complement){
    m.r.ckks->evaluator.negate_inplace(sum);
    sum=m.addc(std::move(sum),1.);
  }
  detail::ScaledPolynomialCipher state{std::move(sum),denominator_gain};
  auto correction=[&]{
    auto value=state.value;
    if(track_complement)m.r.ckks->evaluator.negate_inplace(value);
    return detail::ScaledPolynomialCipher{
      m.addc(std::move(value),(track_complement?2.:1.)*state.gain),state.gain};
  };
  const double inverse_scale=std::ldexp(1.,std::max(0,iterations-4));
  detail::ScaledPolynomialCipher inverse{
    m.scale(correction().value,1./(inverse_scale*denominator_gain)),1.};
  auto refresh=[&](detail::ScaledPolynomialCipher& value,double bound){
    if(value.value.coeff_modulus_size()>4)return;
    value.value=m.scale(std::move(value.value),1./value.gain);
    value.gain=1.;
    detail::column_bounded_refresh(m,value.value,bound);
  };
  for(int i=0;i<iterations;++i){
    refresh(state,2.);
    refresh(inverse,std::max(1.,std::ldexp(1.,i+1)/inverse_scale));
    // Tracking 1 - error keeps a small denominator away from cancellation near one.
    state=detail::scale_aware_product(m,state,track_complement?correction():state);
    inverse=detail::scale_aware_product(m,inverse,correction());
  }
  refresh(inverse,std::max(1.,std::ldexp(1.,iterations+1)/inverse_scale));
  return {std::move(inverse),inverse_scale};
}

}
Cipher nexus_batch::gelu(Runtime& r,Cipher input){
  Math m{r};m.prime_scale=true;
  auto normalized=m.scale(input,1./gelu_sign_divisor);
  auto sign=[&](double shift){
    auto gate=m.addc(normalized,shift/gelu_sign_divisor);
    for(const auto* coefficients:{&sign_g,&sign_f}){
      for(int i=0;i<2;++i)gate=sign_odd9(m,std::move(gate),*coefficients);
      gate=m.scale(std::move(gate),.5);
    }
    if(gate.coeff_modulus_size()<=6)detail::column_bounded_refresh(m,gate,1.);
    return gate;
  };
  auto low=sign(4.),mid=sign(1.95),high=sign(-3.);
  // The outer branches can be large; bound only the input of the two middle polynomials.
  auto clip=detail::column_product(m,input,m.sub(low,high));
  clip=m.addc(m.add(std::move(clip),m.add(m.scale(low,4.),m.scale(high,3.))),-.5);
  if(clip.coeff_modulus_size()<=10)detail::column_bounded_refresh(m,clip,8.);
  auto first=detail::column_product(m,m.poly(clip,gelu_lower),m.sub(low,mid));
  auto second=detail::column_product(m,m.poly(std::move(clip),gelu_upper),m.sub(mid,high));
  auto tail=detail::column_product(m,std::move(input),m.addc(high,.5));
  return m.add(m.add(std::move(first),std::move(second)),std::move(tail));
}

Cipher nexus_batch::softmax(Runtime& r,Cipher scores,std::vector<double> mask,
                            double shift,double divisor,int inverse_iterations){
  TORCH_CHECK(mask.size()==slots&&std::isfinite(shift)&&std::isfinite(divisor)&&divisor>0.&&
              inverse_iterations>=0&&inverse_iterations<=24,"Invalid NEXUS Softmax parameters");
  Math m{r};m.prime_scale=true;
  const bool diagnostic=std::getenv("CIPHEROPBENCH_NEXUS_SOFTMAX_DIAGNOSTIC");
  const auto ideal_inverse=[&](const torch::Tensor& denominator){
    return (1.-(1.-denominator).pow(std::exp2(inverse_iterations+1)))/denominator;
  };
  torch::Tensor reference_exp,reference_padding,reference_sum,reference_inverse;
  if(diagnostic){
    const auto keep=k::tensor_from_vector(mask,{2,128,128});
    reference_exp=(1.+(r.decrypt_slots(scores)-shift)/256.).pow(256)*keep/divisor;
    reference_padding=1.-keep.select(-1,0).unsqueeze(-1);
    reference_sum=reference_exp.sum(-1,true)+reference_padding;
    reference_inverse=ideal_inverse(reference_sum);
  }
  auto observe=[&](const char* stage,const Cipher& value,const torch::Tensor& expected,double factor=1.)
      ->torch::Tensor{
    if(!diagnostic)return {};
    auto decoded=r.decrypt_slots(value)*factor;
    const auto error=(decoded-expected).abs();
    std::cerr<<"NEXUS Softmax "<<stage<<" levels="<<value.coeff_modulus_size()
             <<" max_error="<<error.max().item<double>()<<" mae="<<error.mean().item<double>()<<'\n';
    return decoded;
  };
  // Squaring on a bootstrap prime discards about 11 extra scale bits before the next square.
  if(scores.coeff_modulus_size()>k::L+1)
    r.ckks->evaluator.mod_switch_to_inplace(scores,k::chain_idx(k::L+1));
  const int gain_bits=softmax_exponential_gain_bits(inverse_iterations,divisor);
  const double exponential_gain=std::ldexp(1.,gain_bits);
  const double denominator_gain=std::ldexp(1.,softmax_denominator_gain_bits(gain_bits));
  // Keep small values amplified through row rotations and carry the gain into the reciprocal.
  const double base_gain=std::exp2(gain_bits/256.);
  detail::ScaledPolynomialCipher exponent{
    m.addc(m.scale(std::move(scores),base_gain/256),base_gain*(1.-shift/256)),1.};
  for(int i=0;i<8;++i)exponent=detail::scale_aware_product(m,exponent,exponent);
  TORCH_CHECK(exponent.gain>.5&&exponent.gain<2.,"NEXUS exponential scale drift");
  Vec padding(slots);
  for(int row=0;row<slots;row+=128){
    std::fill_n(padding.begin()+row,128,denominator_gain*(1.-mask[row]));
    for(int j=row;j<row+128;++j)mask[j]/=divisor*exponent.gain;
  }
  auto e=m.pm(std::move(exponent.value),mask);
  const auto decoded_exp=observe("exponentials",e,reference_exp,1./exponential_gain);
  auto sum=m.row_sum(e,denominator_gain/exponential_gain);
  auto pt=k::CKKSEncode(std::move(padding),r.ckks,&sum);
  r.ckks->evaluator.add_plain_inplace(sum,pt);
  const auto decoded_sum=observe("denominator",sum,reference_sum,1./denominator_gain);
  auto [inverse,scale]=scaled_goldschmidt(m,std::move(sum),inverse_iterations,true,denominator_gain);
  const auto decoded_inverse=observe("reciprocal",inverse.value,reference_inverse,scale/inverse.gain);
  auto probability=detail::scale_aware_product(m,{std::move(e),1.},std::move(inverse));
  auto output=m.scale(std::move(probability.value),scale/(probability.gain*exponential_gain));
  if(diagnostic){
    const auto reference=reference_exp*reference_inverse;
    const auto decoded=observe("probability",output,reference);
    const auto worst=(decoded-reference).abs().flatten().argmax().item<int64_t>();
    const auto exp_only=decoded_exp*ideal_inverse(decoded_exp.sum(-1,true)+reference_padding);
    const auto with_row_sum=decoded_exp*ideal_inverse(decoded_sum);
    const auto with_inverse=decoded_exp*decoded_inverse;
    auto at_worst=[&](const torch::Tensor& value){return value.expand_as(reference).flatten()[worst].item<double>();};
    std::cerr<<"NEXUS Softmax probability decomposition expected="<<at_worst(reference)
             <<" after_exp="<<at_worst(exp_only)<<" after_row_sum="<<at_worst(with_row_sum)
             <<" after_inverse="<<at_worst(with_inverse)<<" after_product="<<at_worst(decoded)
             <<" reference_denominator="<<at_worst(reference_sum)<<" decoded_denominator="<<at_worst(decoded_sum)<<'\n';
  }
  return output;
}

LiteratureOperators::LiteratureOperators(Runtime& r,std::string method,bool domains):TransformerOperators(r),method_(std::move(method)),published_domains_(domains){TORCH_CHECK(method_=="euston"||method_=="thor","Unknown literature method");}
CipherList LiteratureOperators::linear(CipherList x,const LinearWeights& w){
  TORCH_CHECK(x.size()==3&&w.weight.sizes()==torch::IntArrayRef({768,768}),"Common linear contract requires 128x768 @ 768x768");Math m{runtime_};CipherList out;
  runtime_.record(method_=="euston"?"euston_ccpumm_with_layout":"thor_complex_pcmm_bsgs16_with_layout",[&]{
    if(method_=="euston"){for(auto& ct:x)m.need(ct,4);out=m.cpmm(m.columns(x,range(768),4),w);return;}
    out=detail::thor_complex_pcmm(m,std::move(x),w);
  },x[0].chain_index());return out;
}
QKV LiteratureOperators::project_qkv(CipherList x,const LayerWeights& w){
  auto q=linear(x,w.q),key=linear(x,w.key),v=linear(x,w.v);Math m{runtime_};QKV out;
  for(int p=0;p<6;++p){auto [a,b]=pairs[p];
    auto map=[=](int i){int h=i/half,t=(i%half)/128,c=i%128,head=h?b:a;if(c>=64)return std::tuple{-1,0,0.};int feature=head*64+c;return std::tuple{feature/256,((feature/128)%2)*half+t*128+feature%128,1.};};
    out.q[p]=m.route(q,map);out.key[p]=m.route(key,map);
    out.v[p]=m.route(v,[=](int i){int t=(i%half)/128,c=i%128,feature=(c<64?a:b)*64+c%64;return std::tuple{feature/256,((feature/128)%2)*half+t*128+feature%128,1.};});
  }return out;
}
Cipher LiteratureOperators::qk(Cipher q,Cipher key){Math m{runtime_};m.prime_scale=method_=="thor";Cipher out;
  runtime_.record(method_=="euston"?"euston_cccrmm_with_layout":"thor_complex_ccmm_bsgs_lazy_with_layout",[&]{
    if(method_=="thor"){q=m.scale(q,.5);key=m.scale(key,.5);auto result=detail::thor_complex_ccmm(m,m.lower(q,false),m.lower(key,true));out=m.row_diagonal(m.unlower(result,false),false,.5);return;}
    std::vector<int> indices;for(int h=0;h<2;++h)for(int c=0;c<64;++c)indices.push_back(h*128+c);
    auto qs=m.columns({q},indices),ks=m.columns({key},indices);CipherList columns;
    for(int h=0;h<2;++h){
      auto head=euston_qk_columns(m,std::span<const Cipher>(qs).subspan(h*64,64),std::span<const Cipher>(ks).subspan(h*64,64),1);
      columns.insert(columns.end(),std::make_move_iterator(head.begin()),std::make_move_iterator(head.end()));
    }
    out=m.pack_columns(columns,256,0.125)[0];
  },q.chain_index(),&out);return out;
}
CipherList LiteratureOperators::batch_qk(const CipherList& q,const CipherList& k){
  TORCH_CHECK(method_=="euston","Batched diagonal QK requires Euston");
  Math m{runtime_};
  auto scores=euston_qk_columns(m,q,k,256);
  for(auto& score:scores)score=m.scale(std::move(score),.125);
  return scores;
}
Cipher LiteratureOperators::av(Cipher probabilities,Cipher value){Math m{runtime_};m.prime_scale=method_=="thor";Cipher out;
  runtime_.record(method_=="euston"?"euston_cdccmm_with_layout":"thor_complex_av_bsgs_lazy_with_layout",[&]{
    if(method_=="thor"){
      auto probability_rows=m.row_diagonal(probabilities,true);
      // Normalize small probabilities before real bootstrap; compensate after CC-MM.
      m.integer_scale(probability_rows,512);
      auto p=m.lower(probability_rows,false);
      m.integer_scale(value,4);
      auto v=m.route({value},[](int i){int h=i/half,d=(i%half)/128,t=i%128;if((h==0&&t>=64)||(h==1&&t<64))return std::tuple{-1,0,0.};return std::tuple{0,h*half+((d+t)%128)*128+t,1.};});
      out=m.scale(m.unlower(detail::thor_complex_ccmm(m,p,v),false),1./2048);return;
    }
    auto p=m.columns({probabilities},range(256));std::vector<int> indices;for(int h=0;h<2;++h)for(int c=0;c<64;++c)indices.push_back(h*128+h*64+c);
    auto v=m.columns({value},indices);CipherList result(256);
    for(int h=0;h<2;++h){
      auto head=euston_av_columns(m,std::span<const Cipher>(p).subspan(h*128,128),std::span<const Cipher>(v).subspan(h*64,64),1);
      for(int c=0;c<128;++c)result[h*128+c]=c/64==h?std::move(head[c%64]):m.zero(v[0]);
    }
    out=m.pack_columns(result,256)[0];
  },probabilities.chain_index(),&out);return out;
}
CipherList LiteratureOperators::batch_av(const CipherList& p,const CipherList& v){
  TORCH_CHECK(method_=="euston","Batched diagonal AV requires Euston");
  Math m{runtime_};
  return euston_av_columns(m,p,v,256);
}
CipherList LiteratureOperators::batch_softmax(CipherList scores,const torch::Tensor& valid,
                                               const EustonSoftmaxProfile& profile){
  constexpr int stride=256,tokens=128;
  TORCH_CHECK(method_=="euston"&&published_domains_&&scores.size()==tokens&&
              valid.dim()==2&&valid.size(1)==tokens&&valid.size(0)>0&&valid.size(0)<=stride,
              "Expected Euston diagonal-packed batch and token mask");
  const auto [shift,divisor,tau,iterations]=profile;
  TORCH_CHECK(std::isfinite(shift)&&std::isfinite(divisor)&&divisor>0&&
              tau>=0&&tau<=3&&iterations>=0&&iterations<=24,"Invalid public Euston Softmax profile");
  Math m{runtime_};m.prime_scale=true;
  const auto mask=valid.to(torch::kCPU).to(torch::kBool).contiguous();
  const auto active=mask.accessor<bool,2>();
  Vec keep(slots);
  for(int t=0;t<tokens;++t)for(int b=0;b<mask.size(0);++b)keep[t*stride+b]=active[b][t];

  const double reduction=std::ldexp(1.,-tau);
  for(int d=0;d<tokens;++d){
    auto z=m.addc(m.scale(std::move(scores[d]),reduction),-shift*reduction);
    if(z.coeff_modulus_size()<=12)detail::column_bounded_refresh(m,z,8.);
    detail::ScaledPolynomialCipher p{m.poly(std::move(z),literature::EUSTON_EXP),1.};
    if(p.value.coeff_modulus_size()<=size_t(tau+2))detail::column_bounded_refresh(m,p.value,2.);
    for(int i=0;i<tau;++i)p=detail::scale_aware_product(m,p,p);
    // Fold the public inverse normalization into the diagonal mask.
    Vec pair(slots);
    for(int j=0;j<slots;++j)pair[j]=keep[j]*keep[(j+d*stride)%slots]/(p.gain*divisor);
    scores[d]=m.pm(std::move(p.value),pair);
  }
  auto sum=scores[0];
  for(int d=1;d<tokens;++d)sum=m.add(std::move(sum),scores[d]);
  Vec padding(slots);
  for(int j=0;j<slots;++j)padding[j]=1.-keep[j];
  auto dummy=k::CKKSEncode(std::move(padding),runtime_.ckks,&sum);
  runtime_.ckks->evaluator.add_plain_inplace(sum,dummy);
  auto [inverse,inverse_scale]=scaled_goldschmidt(m,std::move(sum),iterations);
  for(auto& score:scores){
    if(score.coeff_modulus_size()<=2)detail::column_bounded_refresh(m,score,2.);
    auto probability=detail::scale_aware_product(m,{std::move(score),1.},inverse);
    score=m.scale(std::move(probability.value),inverse_scale/probability.gain);
  }
  return scores;
}

Cipher LiteratureOperators::batch_gelu(Cipher input){
  return batch_gelu(std::move(input),16.);
}

Cipher LiteratureOperators::batch_gelu(Cipher input,double input_bound){
  TORCH_CHECK(method_=="euston"&&published_domains_&&std::isfinite(input_bound)&&input_bound>=1.&&input_bound<=64.,
              "Batched Euston GELU requires a public sign interval");
  Math m{runtime_};m.prime_scale=true;
  auto keep=[&](Cipher& value,size_t levels,double bound){
    if(value.coeff_modulus_size()<=levels)column_bounded_refresh(m,value,bound);
  };
  auto product=[&](Cipher a,Cipher b,double bound_a,double bound_b){
    keep(a,3,bound_a);keep(b,3,bound_b);
    return column_product(m,std::move(a),std::move(b));
  };
  auto sign=m.scale(input,1./input_bound);
  for(const auto* poly:{&sign_g,&sign_f})for(int i=0;i<3;++i)sign=sign_odd9(m,std::move(sign),*poly);
  auto exponential=[&](Cipher x,double coefficient){
    keep(x,3,2.*input_bound);
    x=m.scale(std::move(x),coefficient/8.);
    keep(x,10,8.);x=m.poly(std::move(x),literature::EUSTON_EXP);
    for(int i=0;i<3;++i){keep(x,3,4.);x=column_product(m,x,x);}
    return x;
  };
  auto absolute=product(input,sign,input_bound,2.);
  auto error=m.scale(exponential(absolute,-1.702),-1.);
  auto inverse=m.addc(error,1.);
  for(int i=0;i<3;++i){
    keep(error,4,2.);error=column_product(m,error,error);
    inverse=product(std::move(inverse),m.addc(error,1.),2.,2.);
  }
  auto rectified=exponential(product(m.addc(sign,-1.),absolute,2.,input_bound),.851);
  auto gelu=product(absolute,std::move(inverse),input_bound,2.);
  auto factor=product(std::move(sign),std::move(rectified),2.,2.);
  return product(std::move(gelu),std::move(factor),input_bound,2.);
}

CipherList LiteratureOperators::batch_layer_norm(CipherList x,torch::Tensor weight,torch::Tensor bias,
                                                const std::vector<double>& slot_mask,double divisor){
  constexpr int width=768;
  TORCH_CHECK(method_=="euston"&&published_domains_&&x.size()==width&&weight.numel()==width&&
              bias.numel()==width&&slot_mask.size()==slots&&std::isfinite(divisor)&&divisor>0.,
              "Expected column-packed Euston LayerNorm and a public variance divisor");
  Math m{runtime_};m.prime_scale=true;
  auto mean=x[0];
  for(int j=1;j<width;++j)mean=m.add(std::move(mean),x[j]);
  mean=m.scale(std::move(mean),1./width);
  Cipher squared;
  for(int j=0;j<width;++j){
    x[j]=m.sub(std::move(x[j]),mean);
    auto term=detail::column_product(m,x[j],x[j]);
    squared=j?m.add(std::move(squared),std::move(term)):std::move(term);
  }
  auto t=m.addc(m.scale(std::move(squared),1./divisor),-1.);
  auto t2=detail::column_product(m,t,t),t3=detail::column_product(m,t2,t);
  auto root=m.add(m.addc(m.scale(t,.5),1.),m.add(m.scale(t2,-.125),m.scale(t3,.0625)));
  auto [inverse,inverse_scale]=scaled_goldschmidt(m,m.scale(std::move(root),.5),6);
  const double normalization=std::sqrt(width/divisor)*inverse_scale*.5;
  for(int j=0;j<width;++j){
    auto product=detail::scale_aware_product(m,{std::move(x[j]),1.},inverse);
    const double gamma=weight[j].item<double>()*normalization/product.gain,beta=bias[j].item<double>();
    Vec scale(slots),offset(slots);
    for(int i=0;i<slots;++i){scale[i]=slot_mask[i]*gamma;offset[i]=slot_mask[i]*beta;}
    x[j]=m.pm(std::move(product.value),scale);
    auto pt=k::CKKSEncode(std::move(offset),runtime_.ckks,&x[j]);
    runtime_.ckks->evaluator.add_plain_inplace(x[j],pt);
  }
  return x;
}
CipherList LiteratureOperators::gelu(CipherList x){
  if(runtime_.calibration.enabled)return detail::calibrated_gelu(runtime_,std::move(x),layer_);
  Math m{runtime_};m.prime_scale=published_domains_;CipherList out;
  for(auto& ct:x)runtime_.record(method_=="euston"?"euston_sign_uniform_sigmoid_gelu":"thor_composed_31_27_gelu",[&]{
    if(method_=="thor"){auto p=m.chebyshev(m.scale(ct,1./64),literature::THOR_GELU_P1);auto coefficients=literature::THOR_GELU_P2;for(auto& c:coefficients)c*=0.5;auto y=m.addc(m.chebyshev(p,coefficients),0.5);out.push_back(m.mul(ct,y));return;}
    Vec g{0,5850./1024,0,-34974./1024,0,97015./1024,0,-113492./1024,0,46623./1024};
    Vec f{0,315./128,0,-420./128,0,378./128,0,-180./128,0,35./128};
    auto sign=m.scale(ct,published_domains_?1./16:1./6);for(int i=0;i<3;++i)sign=m.poly(sign,g);for(int i=0;i<3;++i)sign=m.poly(sign,f);
    int tau=published_domains_?3:1;
    auto a=m.mul(ct,sign),den=m.addc(m.exp_euston(m.scale(a,-1.702),tau),1.);auto z=m.mul(a,m.inverse(den,1.,3));
    auto correction=m.exp_euston(m.scale(m.mul(m.addc(sign,-1),a),0.851),tau);out.push_back(m.mul(m.mul(z,sign),correction));
  },ct.chain_index());return out;
}
Cipher LiteratureOperators::softmax(Cipher x,const k::FlatVec& mask){
  if(runtime_.calibration.enabled&&method_!="thor")return detail::calibrated_softmax(runtime_,x,mask,layer_);
  TORCH_CHECK(mask.size()==slots,"Softmax mask contract mismatch");Math m{runtime_};m.prime_scale=true;Cipher out;
  runtime_.record(method_=="euston"?"euston_exp_inverse_common_row_reduction":"thor_adaptive_square_normalize",[&]{
    if(method_=="euston"){
      Vec valid(slots);for(int i=0;i<slots;++i)valid[i]=cuCreal(mask[i])==0?1.:0.;
      auto values=m.pm(m.exp_euston(x,published_domains_?3:1),valid),sum=m.row_sum(values);
      out=m.mul(values,m.inverse(sum,published_domains_?128.:32.,published_domains_?10:4));return;
    }
    bool wide=layer_==2;double low=wide?-70.:-27.2493,high=wide?70.:21.72692;int repeats=wide?2:1,rate=8;
    auto coefficients=wide?literature::THOR_EXP2:literature::THOR_EXP1;
    double epsilon=published_domains_?1./512:std::exp2(wide?-18:-11),first_maximum=1.;
    if(published_domains_){
      double z0=-(low+high)/2/(rate*2*(1<<repeats)),p0=0.;
      for(auto it=coefficients.rbegin();it!=coefficients.rend();++it)p0=p0*z0+*it;
      TORCH_CHECK(std::isfinite(p0)&&p0>0,"Invalid public exponential normalization");
      for(auto& coefficient:coefficients)coefficient/=p0;
      first_maximum=256.;
    }
    auto y=m.square(m.poly(m.scale(m.addc(x,-(low+high)/2),1./(rate*2*(1<<repeats))),coefficients));
    Vec valid(slots);for(int i=0;i<slots;++i)valid[i]=cuCreal(mask[i])==0?1.:0.;y=m.pm(y,valid);
    for(int i=0;i<=repeats;++i){
      auto sum=m.row_sum(y);double alpha=i==repeats?0.001:0.01;
      double maximum=i==0?first_maximum:1.;auto [inv,inverse_factor]=m.adaptive_inverse(sum,maximum,epsilon,alpha);
      if(published_domains_){double y_bound=i==0?first_maximum/256:1.01;double left=std::sqrt(1.01/(maximum*epsilon*y_bound));y=m.mul(m.scale(y,left),m.scale(inv,inverse_factor/left));}
      else y=m.mul(m.scale(y,inverse_factor),inv);
      while(epsilon<1-alpha){double k=2/(1+epsilon);epsilon=k*epsilon*(2-k*epsilon);}
      if(i<repeats){
        // Refresh before squaring, while probabilities have not shrunk to the noise floor.
        m.need(y,3);y=m.square(y);epsilon/=256.;
      }
    }out=std::move(y);
  },x.chain_index(),&out);return out;
}
CipherList LiteratureOperators::layer_norm(CipherList x,torch::Tensor weight,torch::Tensor bias){
  if(runtime_.calibration.enabled)return detail::calibrated_norm(runtime_,std::move(x),weight,bias,layer_,norm_kind_);
  TORCH_CHECK(x.size()==3&&weight.numel()==768&&bias.numel()==768,"LayerNorm common contract mismatch");Math m{runtime_};m.prime_scale=true;CipherList out;
  runtime_.record(method_=="euston"?"euston_taylor_sqrt_goldschmidt_ln":"thor_adaptive_inverse_sqrt_ln",[&]{
    if(method_=="euston"){
      auto mean=m.scale(m.hidden_sum(x),1./768);CipherList centered,squares;
      for(auto& ct:x){centered.push_back(m.sub(ct,mean));squares.push_back(m.square(centered.back()));}
      auto d=m.addc(m.scale(m.hidden_sum(squares),1./1024),-1.);
      auto root=m.add(m.addc(m.scale(d,.5),1.),m.scale(m.square(d),-.125));
      auto inv=m.inverse(root,2.,4);
      for(auto& ct:centered)out.push_back(m.scale(m.mul(ct,inv),sqrt(768./1024)));
    }else{
      auto mean=m.scale(m.hidden_sum(x),1./768);CipherList centered;CipherList squares;
      for(auto& c:x){centered.push_back(m.sub(c,mean));squares.push_back(m.square(centered.back()));}
      auto variance=m.addc(m.scale(m.hidden_sum(squares),1./768),1e-5/(768.*768));
      double low=norm_kind_==1?0.15:(layer_==9||layer_==10)?0.75:0.2,high=norm_kind_==1?10.:(layer_==9||layer_==10)?2500.:150.;
      auto inv=m.adaptive_isqrt(variance,high*1.05+1e-5,low/high);for(auto& c:centered)out.push_back(m.mul(c,inv));
    }
    auto weights=k::row_pack_768x1(weight),biases=k::row_pack_768x1(bias);
    for(int i=0;i<3;++i){Vec gamma(slots);for(int j=0;j<slots;++j)gamma[j]=cuCreal(weights[i][j]);out[i]=m.pm(out[i],gamma);auto pt=k::CKKSEncode(biases[i],runtime_.ckks,&out[i]);runtime_.ckks->evaluator.add_plain_inplace(out[i],pt);}
  },x[0].chain_index());return out;
}
CipherList LiteratureOperators::block(CipherList x,const LayerWeights& w,torch::Tensor mask){
  detail::calibrated_mask(runtime_,x,mask);
  Math m{runtime_};auto out=attention(x,w,mask);for(int i=0;i<3;++i)out[i]=m.add(out[i],x[i]);runtime_.refresh(out);
  norm_kind_=1;out=layer_norm(out,w.ln1_weight,w.ln1_bias);runtime_.refresh(out);out=ffn(out,w);
  norm_kind_=2;out=layer_norm(out,w.ln2_weight,w.ln2_bias);runtime_.refresh(out);norm_kind_=1;return out;
}
CipherList LiteratureOperators::encoder(CipherList x,const std::vector<LayerWeights>& weights,torch::Tensor mask){
  for(size_t i=0;i<weights.size();++i){layer_=i;x=block(x,weights[i],mask);if(runtime_.layer_completed)runtime_.layer_completed(i+1,weights.size());}layer_=0;return x;
}
void register_literature_methods(bool domains){
  MethodRegistry::instance().add({"euston","PhantomBootstrapping_common","common_rows_to_euston_columns","8ff251f03ef9ddbd0c21f1828fac6d084945438a",{}},[domains](Runtime& r){return std::make_unique<LiteratureOperators>(r,"euston",domains);});
  MethodRegistry::instance().add({"thor","PhantomBootstrapping_common","common_rows_to_thor_complex_batched_lower_h2","b513c06aaabccc04b560721e35e9a98008428c25",{}},[domains](Runtime& r){return std::make_unique<LiteratureOperators>(r,"thor",domains);});
}
}
