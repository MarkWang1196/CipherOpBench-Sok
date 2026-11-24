#include "cipheropbench/cachemir_gpu.cuh"
#include "cipheropbench/unified_refresh.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>

namespace cipheropbench::cachemir {
GPU::GPU(std::shared_ptr<PhantomContext> context,
         std::shared_ptr<PhantomPublicKey> pk,
         std::shared_ptr<PhantomCKKSEncoder> encoder,
         std::shared_ptr<PhantomRelinKey> relin,
         std::shared_ptr<PhantomGaloisKey> rotations,bool bootstrap){
  if(!context||!pk||!encoder||!relin||!rotations||encoder->slot_count()!=unified_phantom::physical_slots)
    throw std::invalid_argument("Cachemir requires complete public LLaMA geometry");
  const auto& p=context->key_context_data().parms();
  const auto& q=p.coeff_modulus();
  if(p.poly_modulus_degree()!=(size_t(1)<<unified_phantom::log_n)||q.size()!=size_t(unified_phantom::data_primes+unified_phantom::bootstrap_primes+2)||
     p.secret_key_hamming_weight()!=unified_phantom::secret_hamming_weight)
    throw std::invalid_argument("Cachemir unified encryption parameters");
  for(size_t i=0;i<q.size();++i)
    if(q[i].value()!=unified_phantom::coefficient_primes[i]||q[i].bit_count()!=((i>=1&&i<=size_t(unified_phantom::data_primes))?unified_phantom::log_scale:unified_phantom::log_q0))
      throw std::invalid_argument("Cachemir unified modulus chain");
  public_=std::make_shared<nexus::CKKSEvaluator>(context,pk,nullptr,encoder,relin,rotations,scale);
  public_->set_logical_slots(slots);
  if(bootstrap){
    auto destroy=[](::Bootstrapper* b){delete b->mod_reducer->poly_generator;
      delete b->mod_reducer->inverse_poly_generator;delete b->mod_reducer;delete b;};
    bootstrapper_=std::shared_ptr<::Bootstrapper>(new ::Bootstrapper(unified_phantom::bootstrap_log_width,unified_phantom::logical_log_slots,unified_phantom::log_n-1,unified_phantom::data_primes+unified_phantom::bootstrap_primes,scale,25,unified_phantom::bootstrap_modular_degree,2,unified_phantom::bootstrap_inverse_degree,public_),destroy);
    bootstrapper_->prepare_mod_polynomial();std::vector<int> steps{0};
    for(int i=0;i<15;++i)steps.push_back(1<<i);
    bootstrapper_->addLeftRotKeys_Linear_to_vector_3(steps);
    bootstrapper_->slot_vec.push_back(15);bootstrapper_->generate_LT_coefficient_3();
  }
  synchronize();
}
void GPU::synchronize(){auto e=cudaDeviceSynchronize();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void GPU::need(const Cipher& x,int moduli){
  if(x.size()!=2||x.poly_modulus_degree()!=(size_t(1)<<unified_phantom::log_n)||!x.is_ntt_form()||
     x.coeff_modulus_size()<1||x.coeff_modulus_size()>size_t(unified_phantom::working_moduli)||
     (moduli&&x.coeff_modulus_size()!=size_t(moduli))||
     !std::isfinite(x.scale())||std::abs(std::log2(x.scale())-unified_phantom::log_scale)>1e-8)
    throw std::invalid_argument("Cachemir ciphertext metadata: N="+std::to_string(x.poly_modulus_degree())+
      " size="+std::to_string(x.size())+" NTT="+std::to_string(x.is_ntt_form())+
      " moduli="+std::to_string(x.coeff_modulus_size())+" expected="+std::to_string(moduli)+
      " log_scale="+std::to_string(std::log2(x.scale())));
}
Cipher GPU::drop(Cipher x,int moduli){need(x);if(moduli<1||moduli>int(x.coeff_modulus_size()))throw std::invalid_argument("Cannot increase modulus budget");
  while(x.coeff_modulus_size()>size_t(moduli))public_->evaluator.mod_switch_to_next_inplace(x);return x;}
Cipher GPU::rotate(Cipher x,int shift){need(x);shift=(shift%slots+slots)%slots;
  for(int bit=1;bit<slots;bit*=2)if(shift&bit)public_->evaluator.rotate_vector_inplace(x,bit,*public_->galois_keys);return x;}
Cipher GPU::add(Cipher a,Cipher b){need(a);need(b);int level=std::min(a.coeff_modulus_size(),b.coeff_modulus_size());
  a=drop(std::move(a),level);b=drop(std::move(b),level);public_->evaluator.add_inplace(a,b);return a;}
Cipher GPU::add_constant(Cipher x,double value){need(x);if(!std::isfinite(value))throw std::invalid_argument("Nonfinite constant");
  public_->evaluator.add_const_inplace(x,value);return x;}
Cipher GPU::add_plain(Cipher x,const Values& values){need(x);if(values.size()!=slots)throw std::invalid_argument("Public additive plaintext shape");
  for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite public additive plaintext");
  PhantomPlaintext pt;public_->encoder.encode(values,x.chain_index(),x.scale(),pt);
  public_->evaluator.add_plain_inplace(x,pt);return x;
}
Cipher GPU::negate(Cipher x){need(x);public_->evaluator.negate_inplace(x);return x;}
Cipher GPU::multiply_plain(Cipher x,const Values& values){need(x);if(x.coeff_modulus_size()<2||values.size()!=slots)throw std::invalid_argument("Public plaintext multiplication budget/shape");
  for(double v:values)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite public plaintext");
  double q=public_->context->get_context_data(x.chain_index()).parms().coeff_modulus().back().value();
  PhantomPlaintext pt;public_->encoder.encode(values,x.chain_index(),q*scale/x.scale(),pt);
  public_->evaluator.multiply_plain_inplace(x,pt);public_->evaluator.rescale_to_next_inplace(x);
  if(std::abs(x.scale()/scale-1)>1e-12)throw std::runtime_error("Actual plaintext scale alignment failed");
  x.set_scale(scale);return x;
}
Cipher GPU::multiply_constant(Cipher x,double value){return multiply_plain(std::move(x),Values(slots,value));}
Cipher GPU::product(Cipher a,Cipher b){need(a);need(b);int level=std::min(a.coeff_modulus_size(),b.coeff_modulus_size());
  if(level<2)throw std::invalid_argument("Ciphertext multiplication exhausted working chain");
  a=drop(std::move(a),level);b=drop(std::move(b),level);
  double q=public_->context->get_context_data(a.chain_index()).parms().coeff_modulus().back().value();
  if(std::abs(q/scale-1)>1e-3)throw std::invalid_argument("Bootstrap-only prime used for online arithmetic");
  public_->evaluator.multiply_inplace(a,b);public_->evaluator.relinearize_inplace(a,*public_->relin_keys);
  public_->evaluator.rescale_to_next_inplace(a);
  // Common single-level product policy: q differs slightly from the scale.
  // The induced gain is measured explicitly and full-slot numerical gates apply.
  maximum_product_scale_gain_error_=std::max(maximum_product_scale_gain_error_,std::abs(a.scale()/scale-1));
  a.set_scale(scale);return a;
}
Cipher GPU::clean(Cipher x,int rows,double factor){if(rows<1||rows>hidden||!std::isfinite(factor))throw std::invalid_argument("Public feature mask");
  Values mask(slots,0);for(int i=0;i<rows;++i)mask[i*8]=factor;return multiply_plain(std::move(x),mask);}
Cipher GPU::rms(Cipher x,const Values& gamma,double output_factor){need(x,unified_phantom::working_moduli);if(gamma.size()!=hidden||!std::isfinite(output_factor))throw std::invalid_argument("RMS gamma shape");
  auto variance=product(x,x);
  for(int step=8;step<slots;step*=2)variance=add(variance,rotate(variance,step));
  variance=multiply_constant(std::move(variance),1./hidden);
  for(int step:{-1,-2,-4})variance=add(variance,rotate(variance,step));
  auto an=add_constant(std::move(variance),1e-5);Cipher bn;
  // Positive cubic roots from the release's aSOR update, lower bound 1/16.
  constexpr std::array<double,5> roots{2.2857142857142856,1.638006613588337,
    1.1401610106749882,1.0070807942099778,1.000018757650518};
  for(size_t i=0;i<roots.size();++i){double k=roots[i];auto term=add_constant(negate(an),3/k);
    if(i==0)bn=multiply_constant(term,std::pow(k,1.5)/2);
    else bn=product(multiply_constant(std::move(bn),std::pow(k,1.5)/2),term);
    an=product(multiply_constant(std::move(an),std::pow(k,3)/4),product(term,term));
  }
  need(an,unified_phantom::working_moduli-12);need(bn,unified_phantom::working_moduli-11);Values weights(slots,0);for(int i=0;i<hidden;++i)weights[i*8]=gamma[i]*output_factor;
  auto out=product(multiply_plain(std::move(x),weights),std::move(bn));need(out,unified_phantom::working_moduli-12);return out;
}
Cipher GPU::linear(Cipher x,const Weight& weight){need(x);if(!weight||x.coeff_modulus_size()<2)throw std::invalid_argument("Missing public matrix/budget");
  // Release square Linear: preprocessing is essential for its diagonal layout.
  for(int i=1;i<8;i*=2)x=add(x,rotate(x,i*7));
  std::vector<Cipher> babies;for(int j=0;j<16;++j)babies.push_back(rotate(x,64*j));
  Cipher total;
  for(int giant=0;giant<32;++giant){Cipher part;
    for(int baby=0;baby<16;++baby){Values values(slots,0);
      for(int row=0;row<hidden;++row)for(int lane=0;lane<8;++lane){
        int col=(lane+baby*8+giant*128+row)%hidden;
        values[(lane+8*(row+giant*128))%slots]=.5*weight(row,col);
      }
      auto term=multiply_plain(babies[baby],values);
      part=baby?add(std::move(part),std::move(term)):std::move(term);
    }
    part=rotate(std::move(part),giant*1024);
    total=giant?add(std::move(total),std::move(part)):std::move(part);
  }
  for(int i=1;i<8;i*=2)total=add(total,rotate(total,i));
  // Real model matrices: fuse half into public weights, then restore Re without another level.
  Cipher conjugate;public_->evaluator.complex_conjugate(total,*public_->galois_keys,conjugate);
  return add(std::move(total),std::move(conjugate));
}
std::vector<Cipher> GPU::project(const std::vector<Cipher>& inputs,int in_width,int out_width,const Weight& weight){
  if(in_width<1||in_width>intermediate||out_width<1||out_width>intermediate||
     int(inputs.size())!=(in_width+hidden-1)/hidden||!weight)throw std::invalid_argument("Incomplete public matrix geometry");
  std::vector<Cipher> outputs((out_width+hidden-1)/hidden);
  for(int ot=0;ot<int(outputs.size());++ot)for(int it=0;it<int(inputs.size());++it){
    auto tile=[&,ot,it](int row,int col){int r=ot*hidden+row,c=it*hidden+col;
      return r<out_width&&c<in_width?weight(r,c):0.;};
    auto y=linear(inputs[it],tile);outputs[ot]=it?add(std::move(outputs[ot]),std::move(y)):std::move(y);
  }return outputs;
}
Cipher GPU::rope(Cipher x,int position){need(x);if(position<0||position>=8192)throw std::invalid_argument("RoPE position");
  Values co(slots,0),si0(slots,0),si1(slots,0);
  for(int d=0;d<128;d+=2){double theta=position/std::pow(500000.,double(d)/128);
    for(int h=0;h<32;++h){int a=d*256+h*8,b=a+256;co[a]=co[b]=std::cos(theta);si0[a]=std::sin(theta);si1[b]=-std::sin(theta);}}
  return add(multiply_plain(x,co),add(rotate(multiply_plain(x,si0),-256),rotate(multiply_plain(x,si1),256)));
}
Cipher GPU::silu(Cipher x,const Values& coefficients){need(x,unified_phantom::working_moduli);
  if(coefficients.size()!=1024)throw std::invalid_argument("Cachemir SiLU requires all 1024 release coefficients");
  for(double c:coefficients)if(!std::isfinite(c))throw std::invalid_argument("Nonfinite SiLU coefficient");
  x=multiply_constant(std::move(x),1./20);
  std::map<int,Cipher> basis;basis.emplace(1,x);
  std::function<Cipher(int)> t=[&](int n)->Cipher{
    auto it=basis.find(n);if(it!=basis.end())return it->second;
    auto y=product(t(n/2),t((n+1)/2));y=add(y,y);
    y=n%2?add(std::move(y),negate(x)):add_constant(std::move(y),-1.);
    basis.emplace(n,y);return y;
  };
  // Balanced Chebyshev factorization preserves the release polynomial. Leaves
  // use degree31; its extra coefficient level is an explicit backend adaptation.
  std::function<Cipher(const Values&)> evaluate=[&](const Values& c)->Cipher{
    if(c.size()<=32){
      Cipher result;bool first=true;
      for(size_t n=1;n<c.size();++n){Values mask(slots,0);for(int f=0;f<hidden;++f)mask[f*8]=c[n];
        auto term=multiply_plain(t(int(n)),mask);result=first?std::move(term):add(std::move(result),std::move(term));first=false;}
      if(first)result=add(x,negate(x));Values constant(slots,0);for(int f=0;f<hidden;++f)constant[f*8]=c[0];
      return add_plain(std::move(result),constant);
    }
    size_t split=1;while(split*2<c.size())split*=2;
    Values low(c.begin(),c.begin()+split),high(c.size()-split,0.);high[0]=c[split];
    for(size_t j=1;j<high.size();++j){high[j]=2*c[split+j];low[split-j]-=c[split+j];}
    auto upper=evaluate(high);auto lower=evaluate(low);
    return add(product(std::move(upper),t(int(split))),std::move(lower));
  };
  // Quantize only the coefficient tail, with a uniform Chebyshev error bound.
  // On [-20,20], |T_j(x/20)|<=1, so the discarded error is <=1e-13.
  Values quantized=coefficients;long double tail=0;
  while(quantized.size()>2&&tail+std::abs(quantized.back())<=1e-13L){tail+=std::abs(quantized.back());quantized.pop_back();}
  if(quantized.size()>256)throw std::invalid_argument("SiLU coefficient tail exceeds the shared depth budget");
  auto out=evaluate(quantized);need(out);
  if(out.coeff_modulus_size()<size_t(unified_phantom::working_moduli-10))throw std::runtime_error("SiLU consumed more than its bounded polynomial depth");
  return out;
}
Cipher GPU::multiply_integer(Cipher x,int value){need(x);if(value<1||value>65536)throw std::invalid_argument("Integral public amplitude");
  Cipher base=x,result;bool first=true;
  while(value){if(value&1){result=first?base:add(std::move(result),base);first=false;}value>>=1;if(value)base=add(base,base);}
  return result;
}
Cipher GPU::refresh_scaled(Cipher x,double amplitude,int output_moduli,int rows){need(x);
  if(!bootstrapper_||output_moduli<1||output_moduli>unified_phantom::working_moduli||rows<0||rows>hidden)
    throw std::invalid_argument("Public scaled refresh configuration");
  Values mask;if(rows){mask.assign(slots,0);for(int row=0;row<rows;++row)mask[row*8]=1.;}
  auto result=unified_phantom::precision_refresh(*public_,*bootstrapper_,std::move(x),true,amplitude,mask);
  return drop(std::move(result),output_moduli);
}
Cipher GPU::refresh(Cipher x,double amplitude,int output_moduli,int rows){need(x);
  if(!std::isfinite(amplitude)||amplitude<1)throw std::invalid_argument("Public refresh amplitude");
  x=multiply_constant(std::move(x),1/amplitude);
  return refresh_scaled(std::move(x),amplitude,output_moduli,rows);
}

namespace {
Values feature_mask(int stride,int offset=0,double factor=1.){
  Values v(slots,0);for(int i=offset;i<slots;i+=stride)v[i]=factor;return v;
}
Values block_mask(int block){Values v(slots,0);std::fill_n(v.begin()+block*256,256,1.);return v;}
}
Layer::Layer(GPU& gpu,LayerWeights weights):gpu_(gpu),weights_(std::move(weights)){
  if(!weights_.q||!weights_.k||!weights_.v||!weights_.o||!weights_.gate||!weights_.up||!weights_.down||
     weights_.gamma_attention.size()!=hidden||weights_.gamma_ffn.size()!=hidden||weights_.silu_coefficients.size()!=1024)
    throw std::invalid_argument("Complete public LLaMA layer weights and release SiLU coefficients required");
}
void Layer::append(Cache& cache,int position,Cipher k,Cipher v){
  if(position!=cache.next||position<0||position>=8192)throw std::invalid_argument("Noncontiguous encrypted cache append");
  GPU::need(k,6);GPU::need(v,6);int page=position/4096,local=position%4096,block=local/32,lane=local%32;
  auto pending=std::make_shared<Page>();auto old=cache.pages.find(page);
  if(old!=cache.pages.end())*pending=*old->second;
  auto shifted=gpu_.rotate(std::move(k),-lane);auto previous=pending->k.find(block);
  if(previous!=pending->k.end())shifted=gpu_.add(*previous->second,std::move(shifted));
  pending->k[block]=std::make_shared<Cipher>(std::move(shifted));
  auto clean=gpu_.multiply_plain(std::move(v),feature_mask(32));
  for(int giant=0;giant<16;++giant){auto rotated=gpu_.rotate(clean,-giant*8*256-lane);
    for(int baby=0;baby<8;++baby){int diagonal=giant*8+baby;
      auto update=gpu_.multiply_plain(rotated,block_mask((block-baby+128)%128));
      if(pending->v[diagonal])update=gpu_.add(*pending->v[diagonal],std::move(update));
      GPU::need(update,4);pending->v[diagonal]=std::make_shared<Cipher>(std::move(update));
    }
  }
  cache.pages[page]=pending;++cache.next;
}
Layer::Scores Layer::qk(const Cache& cache,Cipher q){GPU::need(q,7);Scores output;
  if(cache.next<1||cache.next>8192)throw std::invalid_argument("Empty encrypted KV cache");
  for(int group=0;group<4;++group){
    auto packed=gpu_.rotate(gpu_.multiply_plain(q,feature_mask(32,group*8)),group*8);
    for(int step=1;step<32;step*=2)packed=gpu_.add(packed,gpu_.rotate(packed,-step));
    for(int page=0;page*4096<cache.next;++page){const auto& p=*cache.pages.at(page);Cipher sum;bool first=true;
      for(int block=0;block<int(p.k.size());++block){auto product=gpu_.product(packed,*p.k.at(block));
        for(int step=1;step<128;step*=2)product=gpu_.add(product,gpu_.rotate(product,step*256));
        auto masked=gpu_.multiply_plain(std::move(product),block_mask(block));
        sum=first?std::move(masked):gpu_.add(std::move(sum),std::move(masked));first=false;
      }
      GPU::need(sum,4);output[group].push_back(gpu_.multiply_constant(std::move(sum),1/std::sqrt(128.)));
    }
  }return output;
}
Layer::Scores Layer::normalize(const Cache& cache,const Scores& scores,int& calls){Scores output;
  for(int group=0;group<4;++group){std::vector<Cipher> exponentials;Cipher denominator;bool first=true;
    if(scores[group].size()!=cache.pages.size())throw std::invalid_argument("Incomplete attention score pages");
    for(size_t page=0;page<scores[group].size();++page){GPU::need(scores[group][page],3);
      auto x=gpu_.refresh(scores[group][page],unified_phantom::refresh_amplitude,12);++calls;
      x=gpu_.add_constant(std::move(x),-2.5);x=gpu_.multiply_constant(std::move(x),1./128);x=gpu_.add_constant(std::move(x),1.);
      for(int i=0;i<7;++i)x=gpu_.product(x,x);GPU::need(x,4);
      Values valid(slots);for(int i=0;i<slots;++i){int token=int(page)*4096+(i/256)*32+i%32;valid[i]=token<cache.next?1.:0.;}
      x=gpu_.multiply_plain(std::move(x),valid);GPU::need(x,3);exponentials.push_back(x);
      auto sum=x;for(int step=1;step<32;step*=2)sum=gpu_.add(sum,gpu_.rotate(sum,step));
      for(int step=1;step<128;step*=2)sum=gpu_.add(sum,gpu_.rotate(sum,step*256));
      sum=gpu_.multiply_plain(std::move(sum),feature_mask(32));
      for(int step=1;step<32;step*=2)sum=gpu_.add(sum,gpu_.rotate(sum,-step));
      denominator=first?std::move(sum):gpu_.add(std::move(denominator),std::move(sum));first=false;
    }
    double alpha=8./cache.next;denominator=gpu_.multiply_constant(std::move(denominator),alpha/unified_phantom::refresh_amplitude);GPU::need(denominator,1);
    denominator=gpu_.refresh_scaled(std::move(denominator),unified_phantom::refresh_amplitude,13);++calls;
    auto residual=gpu_.add_constant(gpu_.negate(std::move(denominator)),1.);auto inverse=gpu_.add_constant(residual,1.);
    for(int i=0;i<9;++i){residual=gpu_.product(residual,residual);inverse=gpu_.product(std::move(inverse),gpu_.add_constant(residual,1.));}
    GPU::need(inverse,3);
    for(auto& exp:exponentials){auto probability=gpu_.multiply_constant(gpu_.product(exp,inverse),alpha/unified_phantom::refresh_amplitude);GPU::need(probability,1);
      output[group].push_back(gpu_.refresh_scaled(std::move(probability),unified_phantom::refresh_amplitude,6));++calls;}
  }return output;
}
Cipher Layer::av(const Cache& cache,const Scores& probabilities){Cipher total;bool first_group=true;
  for(int group=0;group<4;++group){Cipher accumulated;bool first_page=true;
    if(probabilities[group].size()!=cache.pages.size())throw std::invalid_argument("Incomplete probability pages");
    for(size_t page=0;page<probabilities[group].size();++page){GPU::need(probabilities[group][page],6);
      std::vector<Cipher> babies;for(int j=0;j<8;++j)babies.push_back(gpu_.rotate(probabilities[group][page],j*256));
      Cipher sum;bool first_giant=true;
      for(int giant=0;giant<16;++giant){Cipher part;
        for(int baby=0;baby<8;++baby){auto term=gpu_.product(babies[baby],*cache.pages.at(int(page))->v[giant*8+baby]);
          part=baby?gpu_.add(std::move(part),std::move(term)):std::move(term);}
        auto aligned=gpu_.rotate(std::move(part),giant*8*256);
        sum=first_giant?std::move(aligned):gpu_.add(std::move(sum),std::move(aligned));first_giant=false;
      }
      accumulated=first_page?std::move(sum):gpu_.add(std::move(accumulated),std::move(sum));first_page=false;
    }
    for(int step=1;step<32;step*=2)accumulated=gpu_.add(accumulated,gpu_.rotate(accumulated,step));
    auto aligned=gpu_.rotate(gpu_.multiply_plain(std::move(accumulated),feature_mask(32)),-group*8);
    total=first_group?std::move(aligned):gpu_.add(std::move(total),std::move(aligned));first_group=false;
  }GPU::need(total,2);return total;
}
LayerResult Layer::apply(Cipher input,int position){return apply(std::move(input),position,true);}
LayerResult Layer::apply(Cipher input,int position,bool diagnostics){GPU::need(input,13);
  if(position!=cache_.next||position<0||position>=8192)throw std::invalid_argument("Invalid LLaMA layer position");
  Cache pending=cache_;LayerResult result;
  auto timed=[&](const std::string& name,const std::function<void()>& fn){if(diagnostics)GPU::synchronize();auto t=diagnostics?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};try{fn();if(diagnostics)GPU::synchronize();}catch(const std::exception& e){throw std::runtime_error("Cachemir stage "+name+": "+e.what());}
    if(diagnostics)result.stage_ms[name]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();};
  Cipher norm,restored,q,k,v,attention,h,ffn,o_scaled;
  timed("attention_rms",[&]{norm=gpu_.rms(input,weights_.gamma_attention,1/unified_phantom::refresh_amplitude);if(diagnostics)result.trace["attention_rms"]=gpu_.multiply_integer(norm,int(unified_phantom::refresh_amplitude));});
  timed("attention_rms_refresh",[&]{restored=gpu_.refresh_scaled(norm,unified_phantom::refresh_amplitude,13,hidden);++result.bootstrap_calls;if(diagnostics)result.trace["attention_rms_refreshed"]=restored;});
  timed("qkv",[&]{q=gpu_.linear(gpu_.drop(restored,9),weights_.q);k=gpu_.linear(gpu_.drop(restored,8),weights_.k);v=gpu_.linear(gpu_.drop(restored,7),weights_.v);
    if(diagnostics)result.trace["q"]=q;if(diagnostics)result.trace["k"]=k;if(diagnostics)result.trace["v"]=v;});
  timed("rope",[&]{q=gpu_.rope(std::move(q),position);k=gpu_.rope(std::move(k),position);if(diagnostics)result.trace["rope_q"]=q;if(diagnostics)result.trace["rope_k"]=k;});
  timed("cache_append",[&]{append(pending,position,k,v);});
  Scores scores,probabilities;
  timed("qk_scaled",[&]{scores=qk(pending,q);for(int g=0;g<4;++g)for(size_t p=0;p<scores[g].size();++p)if(diagnostics)result.trace["scores/"+std::to_string(g)+"/"+std::to_string(p)]=scores[g][p];});
  timed("normalizer",[&]{probabilities=normalize(pending,scores,result.bootstrap_calls);for(int g=0;g<4;++g)for(size_t p=0;p<probabilities[g].size();++p)if(diagnostics)result.trace["probabilities/"+std::to_string(g)+"/"+std::to_string(p)]=probabilities[g][p];});
  timed("av",[&]{attention=av(pending,probabilities);if(diagnostics)result.trace["av"]=attention;});
  timed("o_projection",[&]{auto w=[&](int row,int physical){return weights_.o(row,(physical%32)*128+physical/32)/unified_phantom::refresh_amplitude;};
    o_scaled=gpu_.linear(attention,w);if(diagnostics)result.trace["o_raw"]=gpu_.multiply_integer(o_scaled,int(unified_phantom::refresh_amplitude));});
  timed("attention_residual_refresh",[&]{auto skip=gpu_.clean(input,hidden,1/unified_phantom::refresh_amplitude);auto sum=gpu_.add(o_scaled,std::move(skip));if(diagnostics)result.trace["attention_residual"]=gpu_.multiply_integer(sum,int(unified_phantom::refresh_amplitude));h=gpu_.refresh_scaled(std::move(sum),unified_phantom::refresh_amplitude,13,hidden);++result.bootstrap_calls;if(diagnostics)result.trace["hidden_after_attention"]=h;});
  timed("ffn_rms",[&]{norm=gpu_.rms(h,weights_.gamma_ffn,1/unified_phantom::refresh_amplitude);if(diagnostics)result.trace["ffn_rms"]=gpu_.multiply_integer(norm,int(unified_phantom::refresh_amplitude));});
  timed("ffn_rms_refresh",[&]{ffn=gpu_.refresh_scaled(norm,unified_phantom::refresh_amplitude,13,hidden);++result.bootstrap_calls;if(diagnostics)result.trace["ffn_input"]=ffn;});
  std::vector<Cipher> gate,up,swiglu;
  timed("gate_up",[&]{
    gate=gpu_.project({gpu_.drop(ffn,3)},hidden,intermediate,weights_.gate);
    up=gpu_.project({gpu_.drop(ffn,3)},hidden,intermediate,weights_.up);
    for(int tile=0;tile<4;++tile){int rows=tile==3?2048:hidden;
      gate[tile]=gpu_.refresh(std::move(gate[tile]),unified_phantom::refresh_amplitude,13,rows);up[tile]=gpu_.refresh(std::move(up[tile]),unified_phantom::refresh_amplitude,13,rows);result.bootstrap_calls+=2;
      if(diagnostics)result.trace["gate/"+std::to_string(tile)]=gate[tile];if(diagnostics)result.trace["up/"+std::to_string(tile)]=up[tile];}
  });
  timed("swiglu",[&]{for(int tile=0;tile<4;++tile){auto activated=gpu_.silu(gate[tile],weights_.silu_coefficients);
    auto y=gpu_.product(std::move(activated),up[tile]);if(diagnostics)result.trace["swiglu/"+std::to_string(tile)]=y;
    int rows=tile==3?2048:hidden;swiglu.push_back(gpu_.drop(gpu_.refresh(std::move(y),unified_phantom::refresh_amplitude,13,rows),3));++result.bootstrap_calls;}
  });
  timed("down_residual_refresh",[&]{auto down=gpu_.project(swiglu,intermediate,hidden,weights_.down);
    auto y=gpu_.clean(std::move(down[0]),hidden,1/unified_phantom::refresh_amplitude);if(diagnostics)result.trace["down"]=gpu_.multiply_integer(y,int(unified_phantom::refresh_amplitude));
    auto skip=gpu_.clean(h,hidden,1/unified_phantom::refresh_amplitude);auto sum=gpu_.add(std::move(y),std::move(skip));if(diagnostics)result.trace["output_residual"]=gpu_.multiply_integer(sum,int(unified_phantom::refresh_amplitude));
    result.output=gpu_.refresh_scaled(std::move(sum),unified_phantom::refresh_amplitude,13,hidden);++result.bootstrap_calls;if(diagnostics)result.trace["output"]=result.output;
  });
  GPU::need(result.output,13);cache_=std::move(pending);return result;
}
}
