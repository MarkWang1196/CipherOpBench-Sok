#pragma once
#include "research_math.cuh"
namespace cipheropbench::detail {
inline Cipher arion_exp_precision(Runtime& r,Cipher x,const std::vector<double>& valid){
 TORCH_CHECK(x.coeff_modulus_size()>=25&&x.scale()==k::scale&&valid.size()==slots,"ARION exp80 requires25 levels, default input scale and a full public mask");
 const double work=std::ldexp(1.,80);Math base{r};base.integer_scale(x,int64_t(1)<<40);x.set_scale(work);
 auto pm=[&](Cipher a,const std::vector<double>& values,double target){TORCH_CHECK(a.coeff_modulus_size()>1,"ARION exp80 exhausted levels");double q=r.ckks->context->get_context_data(a.chain_index()).parms().coeff_modulus().back().value();PhantomPlaintext p;r.ckks->encoder.encode(values,a.chain_index(),q*target/a.scale(),p);r.ckks->evaluator.multiply_plain_inplace(a,p);r.ckks->evaluator.rescale_to_next_inplace(a);a.set_scale(target);return a;};
 auto align=[&](Cipher& a,Cipher& b){TORCH_CHECK(a.scale()==work&&b.scale()==work,"Unexpected exp80 operand scale");auto chain=std::max(a.chain_index(),b.chain_index());if(a.chain_index()!=chain)r.ckks->evaluator.mod_switch_to_inplace(a,chain);if(b.chain_index()!=chain)r.ckks->evaluator.mod_switch_to_inplace(b,chain);};
 auto add=[&](Cipher a,Cipher b){align(a,b);r.ckks->evaluator.add_inplace(a,b);return a;};
 auto sub=[&](Cipher a,Cipher b){align(a,b);r.ckks->evaluator.sub_inplace(a,b);return a;};
 auto addc=[&](Cipher a,double c){PhantomPlaintext p;r.ckks->encoder.encode(std::vector<double>(slots,c),a.chain_index(),a.scale(),p);r.ckks->evaluator.add_plain_inplace(a,p);return a;};
 auto mul=[&](Cipher a,Cipher b){align(a,b);TORCH_CHECK(a.coeff_modulus_size()>=4,"ARION exp80 product exhausted levels");Cipher c;r.ckks->evaluator.multiply(a,b,c);r.ckks->evaluator.relinearize_inplace(c,*r.ckks->relin_keys);r.ckks->evaluator.rescale_to_next_inplace(c);r.ckks->evaluator.rescale_to_next_inplace(c);return pm(std::move(c),std::vector<double>(slots,1.),work);};
 x=addc(pm(std::move(x),std::vector<double>(slots,2./75),work),.6);
 auto p=research_polynomial([](double z){return std::exp(z);},-60.,15.,63);std::map<int,Cipher> basis;basis[1]=x;
 std::function<Cipher(int)> term=[&](int j)->Cipher{auto found=basis.find(j);if(found!=basis.end())return found->second;auto product=mul(term(j/2),term(j%2?j/2+1:j/2));auto y=j%2?sub(add(product,product),x):addc(add(product,product),-1.);basis.emplace(j,y);return y;};
 auto y=addc(pm(x,std::vector<double>(slots,0.),work),p.coefficients[0]);for(int j=1;j<=63;++j)if(p.coefficients[j]!=0.)y=add(y,pm(term(j),std::vector<double>(slots,p.coefficients[j]),work));
 // Restore the common output scale while applying the public row padding mask.
 r.ckks->evaluator.rescale_to_next_inplace(y);return pm(std::move(y),valid,k::scale);
}
}
