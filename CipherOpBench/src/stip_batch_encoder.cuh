namespace {
constexpr int stip_batch=32,stip_width=6,stip_stride=256,stip_features=768,stip_head_width=64;
using StipGroups=std::array<CipherList,stip_width>;

StipGroups stip_group_columns(Math& m,CipherList& input){
 column_ready(m,input,9,12);
 Vec front(slots),wrap(slots);
 for(int i=0;i<slots;++i){
  front[i]=i%stip_stride<(stip_width-1)*stip_batch;
  wrap[i]=i%stip_stride>=(stip_width-1)*stip_batch&&i%stip_stride<stip_width*stip_batch;
 }
 StipGroups groups;
 for(int g=0;g<stip_features/stip_width;++g){
  auto first=input[g*stip_width];
  for(int lane=1;lane<stip_width;++lane)
   first=m.add(first,m.rot(input[g*stip_width+lane],-lane*stip_batch));
  groups[0].push_back(std::move(first));
  for(int r=1;r<stip_width;++r){
   const auto& old=groups[r-1].back();
   groups[r].push_back(m.add(m.pm(m.rot(old,stip_batch),front),
                             m.pm(m.rot(old,-(stip_width-1)*stip_batch),wrap)));
  }
 }
 return groups;
}

template<class Coefficients> Cipher stip_weighted_sum(Math& m,size_t count,const Coefficients& coefficient){
 Cipher sum;
 for(size_t i=0;i<count;++i){
  const auto [source,values]=coefficient(i);
  auto input=source;
  const double prime=m.r.ckks->context->get_context_data(input.chain_index()).parms().coeff_modulus().back().value();
  PhantomPlaintext plain;
  m.r.ckks->encoder.encode(values,input.chain_index(),prime*k::scale/input.scale(),plain);
  Cipher term;m.r.ckks->evaluator.multiply_plain(input,plain,term);
  if(i)m.r.ckks->evaluator.add_inplace(sum,term);else sum=std::move(term);
 }
 m.r.ckks->evaluator.rescale_to_next_inplace(sum);sum.set_scale(k::scale);return sum;
}

void stip_align_groups(Math& m,StipGroups& groups){
 size_t depth=0;
 for(auto& state:groups){column_ready(m,state,4,8);depth=std::max(depth,state[0].chain_index());}
 for(auto& state:groups)for(auto& c:state)m.r.ckks->evaluator.mod_switch_to_inplace(c,depth);
}

std::array<CipherList,2> stip_input_projection(Math& m,const StipGroups& groups,const LinearWeights& weight){
 auto w=weight.weight.accessor<double,2>();
 auto bias=weight.bias.accessor<double,1>();
 std::array<CipherList,2> result;
 for(int f=0;f<stip_head_width;++f){
  auto out=stip_weighted_sum(m,stip_features,[&](size_t i){
   const int g=i/stip_width,r=i%stip_width;
   k::FlatVec values(slots);
   for(int h=0;h<stip_width;++h){
    int input=g*stip_width+(h+r)%stip_width;
    auto value=make_cuDoubleComplex(w[input][h*stip_head_width+f],w[input][(h+stip_width)*stip_head_width+f]);
    for(int t=0;t<128;++t)std::fill_n(values.begin()+t*stip_stride+h*stip_batch,stip_batch,value);
   }
   return std::pair<const Cipher&,k::FlatVec>{groups[r][g],std::move(values)};
  });
  k::FlatVec offset(slots);
  for(int h=0;h<stip_width;++h){
   auto value=make_cuDoubleComplex(bias[h*stip_head_width+f],bias[(h+stip_width)*stip_head_width+f]);
   for(int t=0;t<128;++t)std::fill_n(offset.begin()+t*stip_stride+h*stip_batch,stip_batch,value);
  }
  auto plain=k::CKKSEncode(offset,m.r.ckks,&out);m.r.ckks->evaluator.add_plain_inplace(out,plain);
  result[0].push_back(real_part(m,out));result[1].push_back(imaginary_part(m,out));
 }
 return result;
}

CipherList stip_output_projection(Math& m,std::array<CipherList,2> heads,const LinearWeights& weight,const Vec& valid){
 column_ready(m,heads[0],4,8);column_ready(m,heads[1],4,8);
 for(int j=0;j<stip_head_width;++j)heads[0][j]=m.add(heads[0][j],thor_imaginary(m,heads[1][j]));
 column_ready(m,heads[0],2,8);
 auto w=weight.weight.accessor<double,2>();auto bias=weight.bias.accessor<double,1>();
 CipherList result;
 for(int f=0;f<stip_features;++f){
  auto packed=stip_weighted_sum(m,stip_head_width,[&](size_t j){
   k::FlatVec values(slots);
   for(int h=0;h<stip_width;++h){
    auto value=make_cuDoubleComplex(w[h*stip_head_width+j][f],-w[(h+stip_width)*stip_head_width+j][f]);
    for(int t=0;t<128;++t)std::fill_n(values.begin()+t*stip_stride+h*stip_batch,stip_batch,value);
   }
   return std::pair<const Cipher&,k::FlatVec>{heads[0][j],std::move(values)};
  });
  auto real=real_part(m,packed),out=real;
  for(int h=1;h<stip_width;++h)out=m.add(out,m.rot(real,h*stip_batch));
  result.push_back(m.pm(m.addc(std::move(out),bias[f]),valid));
 }
 return result;
}

Cipher stip_column_root(Math& m,CipherList& x,double divisor){
 column_ready(m,x);
 auto sum=x[0];for(size_t j=1;j<x.size();++j)sum=m.add(sum,x[j]);
 auto mean=m.scale(sum,1./stip_features);
 Cipher squares;
 for(size_t j=0;j<x.size();++j){
  x[j]=m.sub(x[j],mean);auto square=column_product(m,x[j],x[j]);
  squares=j?m.add(squares,square):std::move(square);
 }
 auto t=m.addc(m.scale(squares,1./divisor),-1.);
 return m.add(m.addc(m.scale(t,.5),1.),m.scale(column_product(m,t,t),-.125));
}
}

std::array<CipherList,6> ResearchOperators::batch_stip_prepare(torch::Tensor input,torch::Tensor valid){
 TORCH_CHECK(method_=="stip"&&input.sizes()==torch::IntArrayRef({stip_batch,128,stip_features})&&
             valid.sizes()==torch::IntArrayRef({stip_batch,128}),"Invalid STIP batch input");
 auto prepared=input.to(torch::kCPU).to(torch::kDouble).contiguous();
 auto mask=valid.to(torch::kCPU).to(torch::kBool).contiguous();
 auto values=prepared.accessor<double,3>();auto keep=mask.accessor<bool,2>();
 StipGroups groups;
 for(int r=0;r<stip_width;++r)for(int g=0;g<stip_features/stip_width;++g){
  k::FlatVec packed(slots);
  for(int t=0;t<128;++t)for(int h=0;h<stip_width;++h)for(int b=0;b<stip_batch;++b)
   if(keep[b][t])packed[t*stip_stride+h*stip_batch+b]=make_cuDoubleComplex(values[b][t][g*stip_width+(h+r)%stip_width],0.);
  groups[r].push_back(runtime_.encrypt(std::move(packed),k::chain_idx(9)));
 }
 return groups;
}

std::array<CipherList,6> ResearchOperators::batch_stip_regroup(CipherList& input){
 TORCH_CHECK(method_=="stip"&&input.size()==stip_features,"STIP AMCP requires 768 input columns");
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=stip_stride;
 auto groups=stip_group_columns(m,input);
 stip_align_groups(m,groups);
 return groups;
}

Cipher ResearchOperators::batch_stip_relu(Cipher input){
 TORCH_CHECK(method_=="stip"&&published_domains_,"Expected STIP published ReLU");
 Math m{runtime_};m.prime_scale=true;
 if(input.coeff_modulus_size()<=3)column_bounded_refresh(m,input,16.);
 auto sign=m.scale(input,1./16.);
 for(const auto* coefficients:{&STIP_RELU_0,&STIP_RELU_1}){
  if(sign.coeff_modulus_size()<=9)column_bounded_refresh(m,sign,2.);
  auto value=scale_aware_cheb_normalized(m,std::move(sign),*coefficients);
  sign=m.scale(std::move(value.value),1./value.gain);
 }
 if(sign.coeff_modulus_size()<=2)column_bounded_refresh(m,sign,2.);
 return column_product(m,std::move(input),m.addc(std::move(sign),.5));
}

CipherList ResearchOperators::batch_stip_encoder(CipherList input,const std::array<CipherList,6>& first_projection,
    const std::vector<LayerWeights>& weights,torch::Tensor valid,const std::vector<StipBatchProfile>& profiles){
 TORCH_CHECK(method_=="stip"&&published_domains_&&input.size()==stip_features&&weights.size()==profiles.size()&&
             valid.sizes()==torch::IntArrayRef({stip_batch,128}),"Invalid STIP encoder batch");
 Math m{runtime_};m.prime_scale=true;m.column_token_stride=stip_stride;
 auto mask=valid.to(torch::kCPU).to(torch::kBool).contiguous();
 auto keep=mask.accessor<bool,2>();Vec slots_valid(slots);
 for(int t=0;t<128;++t)for(int b=0;b<stip_batch;++b)slots_valid[t*stip_stride+b]=keep[b][t];
 auto stage=[&](const char* name,const std::function<void()>& run){runtime_.record(std::string("batch.")+name,run);};
 auto mask_columns=[&](CipherList& values){for(auto& c:values)c=m.pm(std::move(c),slots_valid);};
 for(size_t layer=0;layer<weights.size();++layer){
  const auto& w=weights[layer];const auto& p=profiles[layer];
  stage("input_mask",[&]{mask_columns(input);});
  std::array<CipherList,2> q,k,v,heads,scores;
  stage("qkv",[&]{
   StipGroups groups;
   if(layer)groups=batch_stip_regroup(input);
   const auto& projected_input=layer?groups:first_projection;
   q=stip_input_projection(m,projected_input,w.q);k=stip_input_projection(m,projected_input,w.key);v=stip_input_projection(m,projected_input,w.v);
  });
  stage("qk",[&]{for(int g=0;g<2;++g)scores[g]=batch_stip_l2(std::move(q[g]),std::move(k[g]));});
  stage("softmax",[&]{for(auto& diagonal:scores)diagonal=batch_stip_gk(std::move(diagonal),valid);});
  stage("av",[&]{for(int g=0;g<2;++g)heads[g]=batch_stip_av(std::move(scores[g]),std::move(v[g]));});
  stage("attention_ready",[&]{for(auto& head:heads)column_ready(m,head);});
  CipherList hidden;
  stage("wo",[&]{hidden=stip_output_projection(m,std::move(heads),w.out,slots_valid);});
  stage("attention_residual",[&]{for(int j=0;j<stip_features;++j)hidden[j]=m.add(hidden[j],input[j]);});
  input.clear();
  Cipher carried;
  stage("ln1",[&]{
   carried=m.scale(stip_column_root(m,hidden,p.norm1_divisor),1./p.carry_divisor);
   if(carried.coeff_modulus_size()<=5)column_bounded_refresh(m,carried,2.);
   carried=m.pm(std::move(carried),slots_valid);
   const double gamma=std::sqrt(stip_features/p.norm1_divisor)/p.carry_divisor;
   for(int j=0;j<stip_features;++j)
    hidden[j]=m.add(m.scale(hidden[j],gamma*w.ln1_weight[j].item<double>()),m.scale(carried,w.ln1_bias[j].item<double>()));
  });
  auto bias=[&](Cipher value,double weight){return m.add(std::move(value),m.scale(carried,weight));};
  CipherList output(stip_features);
  for(int lo=0;lo<3072;lo+=stip_head_width){
   const int hi=lo+stip_head_width;
   CipherList ff;
   stage("ffn_up",[&]{
    ff=column_cpmm(m,hidden,{w.up.weight.slice(1,lo,hi),torch::zeros(stip_head_width,w.up.bias.options())},3);
    for(int j=0;j<stip_head_width;++j)ff[j]=bias(std::move(ff[j]),w.up.bias[lo+j].item<double>());
   });
   stage("activation",[&]{for(auto& value:ff)value=batch_stip_relu(std::move(value));});
   stage("ffn_down",[&]{
    auto part=column_cpmm(m,ff,{w.down.weight.slice(0,lo,hi),torch::zeros_like(w.down.bias)},2);
    for(int j=0;j<stip_features;++j)output[j]=lo?m.add(output[j],part[j]):std::move(part[j]);
   });
  }
  stage("ffn_residual",[&]{
   for(int j=0;j<stip_features;++j)output[j]=m.add(bias(std::move(output[j]),w.down.bias[j].item<double>()),hidden[j]);
  });
  stage("ln2",[&]{
   auto root=stip_column_root(m,output,p.norm2_divisor);
   auto reciprocal=m.inverse(root,2.,4);
   if(reciprocal.coeff_modulus_size()<=4)column_bounded_refresh(m,reciprocal,4.);
   const double factor=std::sqrt(stip_features/p.norm2_divisor);
   for(int j=0;j<stip_features;++j)
    output[j]=m.addc(m.scale(column_product(m,output[j],reciprocal),factor*w.ln2_weight[j].item<double>()),w.ln2_bias[j].item<double>());
   mask_columns(output);column_ready(m,output);
  });
  input=std::move(output);
  if(runtime_.layer_completed)runtime_.layer_completed(layer+1,weights.size());
 }
 return input;
}
