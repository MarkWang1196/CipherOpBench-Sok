#include "cipheropbench/operators.cuh"
#include "cipheropbench/method_registry.cuh"
#include "cipheropbench/column_methods.cuh"
#include "cipheropbench/components.cuh"
#include "cipheropbench/published.cuh"
#include "cipheropbench/benchmark_validation.cuh"
#include "cipheropbench/literature.cuh"
#include "cipheropbench/extended.cuh"
#include "cipheropbench/research.cuh"
#ifdef CIPHEROPBENCH_NATIVE_BASELINE
#include "bert/bert.cuh"
#endif
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <unistd.h>

namespace {
using namespace cipheropbench;
struct Scope { std::string name, level, unit; bool bootstrap; };
const std::vector<Scope> scopes{
  {"bootstrap","primitive","three hidden ciphertexts, level-1 input",true},
  {"ckks_add","primitive","three hidden ciphertexts, elementwise add",false},
  {"ckks_mul","primitive","three hidden ciphertexts, square + relin + rescale",false},
  {"linear","operator","128x768 @ 768x768",false},
  {"qkv","subgraph","12 heads, three projections with bias",false},
  {"qk","operator","12 heads, QK^T / sqrt(64)",false},
  {"gaussian_scores","operator","12 heads, negative squared distance / 16 for STIP Gaussian attention",true},
  {"softmax","operator","12 heads, includes score and denominator refresh",true},
  {"softmax_shifted","operator","12 heads, supplied row-max-shifted scores; plaintext preparation excluded",true},
  {"av","operator","12 heads, Softmax(QK^T)V",false},
  {"softmax_v","subgraph","12 heads, connected Softmax -> AV",true},
  {"qk_av","subgraph","12 heads, connected QK^T/sqrt(64) -> AV, no Softmax",true},
  {"attention","subgraph","QKV -> QK -> Softmax -> AV -> output projection",true},
  {"gelu","operator","128x768, three ciphertexts",true},
  {"activation","operator","128x3072 FFN preactivation, four encrypted 768-wide tiles; input projection excluded",true},
  {"layernorm","operator","128x768, includes variance refresh",true},
  {"ffn","subgraph","768 -> 3072 -> 768 with GELU and residual",true},
  {"block","block","standard BERT block including both residuals and LayerNorms",true},
  {"encoder","model","connected encoder layers; excludes embedding, pooler, classifier",true}
};
std::string quote(const std::string& s) {
  std::string out="\"";
  for(unsigned char c:s) {
    if(c=='"'||c=='\\') {out+='\\';out+=c;}
    else if(c=='\n') out+="\\n";
    else if(c=='\r') out+="\\r";
    else if(c=='\t') out+="\\t";
    else if(c>=32) out+=c;
  }
  return out+'"';
}
void number(std::ostream& stream,double value) { if(std::isfinite(value)) stream<<std::setprecision(12)<<value; else stream<<"null"; }
torch::Tensor heads(torch::Tensor t) {return t.view({128,12,64}).transpose(0,1).contiguous();}
torch::Tensor pad(torch::Tensor t) {return torch::cat({t,torch::zeros({128,64},torch::kDouble)},1);}
struct Case {
  std::function<void()> evaluate;
  std::function<torch::Tensor()> decode;
  torch::Tensor reference;
};
Case make_case(const Scope& spec,Runtime& r,TransformerOperators& ops,const Fixture& f,const std::string& track,const std::string& method) {
  const auto& w=f.layers[0];
  auto hidden=std::make_shared<CipherList>(r.encrypt_hidden(f.hidden));
  auto output=std::make_shared<CipherList>();
  if(track=="common"&&spec.name=="bootstrap") {
    for(auto& ct:*hidden) r.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(1));
    return {[&r,hidden,output]{*output=*hidden;r.refresh(*output);},[&r,output]{return r.decrypt_hidden(*output);},f.hidden};
  }
  if(track=="common"&&spec.name=="ckks_add") {
    return {[&r,hidden,output]{output->resize(3);for(int i=0;i<3;++i)r.ckks->evaluator.add((*hidden)[i],(*hidden)[i],(*output)[i]);},[&r,output]{return r.decrypt_hidden(*output);},2*f.hidden};
  }
  if(track=="common"&&spec.name=="ckks_mul") {
    return {[&r,hidden,output]{output->resize(3);for(int i=0;i<3;++i){r.ckks->evaluator.square((*hidden)[i],(*output)[i]);r.ckks->evaluator.relinearize_inplace((*output)[i],*r.ckks->relin_keys);r.ckks->evaluator.rescale_to_next_inplace((*output)[i]);}},[&r,output]{return r.decrypt_hidden(*output);},f.hidden.square()};
  }
  if(track=="native") {
#ifdef CIPHEROPBENCH_NATIVE_BASELINE
    auto state=native_state(w);
    if(spec.name=="attention") {
      auto model=std::make_shared<nexus::BertAttention>(r.ckks,r.bootstrapper);
      model->load_state_dict(state,"bert.encoder.layer.0.attention");model->pack_weights();
      auto mask=nexus::convert_mask(f.mask); // Preserve released mask packing.
      auto ref=model->forward(f.hidden.to(torch::kFloat).clone(),f.mask).to(torch::kDouble);
      return {[model,hidden,output,mask]{auto copy=*hidden;*output=model->forward(copy,mask);},[&r,output]{return r.decrypt_hidden(*output);},ref};
    }
    if(spec.name=="ffn") {
      auto model=std::make_shared<nexus::BertMLP>(r.ckks,r.bootstrapper);
      model->load_state_dict(state,"bert.encoder.layer.0");model->pack_weights();
      for(auto& ct:*hidden) r.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(14));
      auto ref=model->forward(f.hidden.to(torch::kFloat)).to(torch::kDouble);
      return {[model,hidden,output]{auto copy=*hidden;*output=model->forward(copy);},[&r,output]{return r.decrypt_hidden(*output);},ref};
    }
    if(spec.name=="block") {
      auto model=std::make_shared<nexus::BertLayer>(r.ckks,r.bootstrapper);
      model->load_state_dict(state,"bert.encoder.layer.0");model->pack_weights();
      auto mask=nexus::convert_mask(f.mask);
      auto ref=model->forward(f.hidden.to(torch::kFloat).clone(),f.mask).to(torch::kDouble);
      return {[model,hidden,output,mask]{auto copy=*hidden;*output=model->forward(copy,mask);},[&r,output]{return r.decrypt_hidden(*output);},ref};
    }
#endif
    throw std::runtime_error("native track supports attention, ffn, block; use native Catch2 tests for isolated kernels");
  }
  if(spec.name=="linear") return {[&ops,hidden,output,&w]{*output=ops.linear(*hidden,w.out);},[&r,output]{return r.decrypt_hidden(*output);},plain_linear(f.hidden,w.out)};
  if(spec.name=="qkv") {
    auto projected=std::make_shared<QKV>();
    return {[&ops,hidden,projected,&w]{*projected=ops.project_qkv(*hidden,w);},[&ops,projected]{return ops.unpack_qkv(*projected);},torch::stack({plain_linear(f.hidden,w.q),plain_linear(f.hidden,w.key),plain_linear(f.hidden,w.v)})};
  }
  if(spec.name=="gelu") {
    for(auto& ct:*hidden) r.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(14));
    return {[&ops,hidden,output]{*output=ops.gelu(*hidden);},[&r,output]{return r.decrypt_hidden(*output);},torch::gelu(f.hidden)};
  }
  if(spec.name=="activation") {
    auto pre=plain_linear(f.hidden,w.up);
    auto tiles=std::make_shared<std::vector<CipherList>>(),activated=std::make_shared<std::vector<CipherList>>(4);
    for(int i=0;i<4;++i){
      tiles->push_back(r.encrypt_hidden(pre.slice(1,i*768,(i+1)*768).contiguous()));
      for(auto& ct:tiles->back())r.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(14));
    }
    return {[&ops,tiles,activated]{for(int i=0;i<4;++i)(*activated)[i]=ops.gelu((*tiles)[i]);},
      [&r,activated]{std::vector<torch::Tensor> parts;for(auto& tile:*activated)parts.push_back(r.decrypt_hidden(tile));return torch::cat(parts,1);},torch::gelu(pre)};
  }
  if(spec.name=="layernorm") {
    for(auto& ct:*hidden) r.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(6));
    return {[&ops,hidden,output,&w]{*output=ops.layer_norm(*hidden,w.ln1_weight,w.ln1_bias);},[&r,output]{return r.decrypt_hidden(*output);},torch::layer_norm(f.hidden,{768},w.ln1_weight,w.ln1_bias,1e-12)};
  }
  if(spec.name=="attention") return {[&ops,hidden,output,&f,&w]{*output=ops.attention(*hidden,w,f.mask);},[&r,output]{return r.decrypt_hidden(*output);},plain_attention(f.hidden,f.mask,w)};
  if(spec.name=="ffn") {
    for(auto& ct:*hidden) r.ckks->evaluator.mod_switch_to_inplace(ct,k::chain_idx(14));
    return {[&ops,hidden,output,&w]{*output=ops.ffn(*hidden,w);},[&r,output]{return r.decrypt_hidden(*output);},plain_ffn(f.hidden,w)};
  }
  if(spec.name=="block") return {[&ops,hidden,output,&f,&w]{*output=ops.block(*hidden,w,f.mask);},[&r,output]{return r.decrypt_hidden(*output);},plain_block(f.hidden,f.mask,w)};
  if(spec.name=="encoder") {
    auto ref=f.hidden;
    for(const auto& layer:f.layers) ref=plain_block(ref,f.mask,layer);
    return {[&ops,hidden,output,&f]{*output=ops.encoder(*hidden,f.layers,f.mask);},[&r,output]{return r.decrypt_hidden(*output);},ref};
  }
  // Isolated head operators receive exact oracle intermediates, encrypted once
  // outside timing. Connected attention never decrypts at these boundaries.
  auto q=heads(plain_linear(f.hidden,w.q)), key=heads(plain_linear(f.hidden,w.key)), value=heads(plain_linear(f.hidden,w.v));
  auto scores=torch::matmul(q,key.transpose(1,2))/8.;
  auto masked=scores.masked_fill(~f.mask.unsqueeze(0),-100.);
  auto probabilities=torch::softmax(masked,-1);
  const bool gaussian=spec.name=="gaussian_scores";
  TORCH_CHECK(!gaussian||method=="stip","Gaussian scores require STIP");
  const bool shifted=spec.name=="softmax_shifted";
  TORCH_CHECK(!shifted||(method!="stip"&&method!="ensi"),"This attention variant is not shift invariant");
  if(shifted)scores=scores-std::get<0>(scores.masked_fill(~f.mask.unsqueeze(0),-INFINITY).max(-1,true));
  auto lhs=std::make_shared<CipherList>(),rhs=std::make_shared<CipherList>(),values=std::make_shared<CipherList>();
  constexpr std::array<std::array<int,2>,6> pairs{{{0,3},{2,1},{4,7},{6,5},{8,11},{10,9}}};
  for(auto [a,b]:pairs) {
    if(spec.name=="qk"||spec.name=="qk_av"||gaussian) {
      if(spec.name=="qk_av")values->push_back(r.encrypt(k::flatten_pack(torch::cat({value[a],value[b]},1)),15));
      lhs->push_back(r.encrypt(k::flatten_pack(pad(q[a]),pad(q[b])),k::chain_idx(5)));
      rhs->push_back(r.encrypt(k::flatten_pack(pad(key[a]),pad(key[b])),k::chain_idx(5)));
    } else {
      auto s=spec.name=="av"?probabilities:scores;
      if(method=="stip"&&spec.name!="av")s=-(q.square().sum(-1,true)+key.square().sum(-1).unsqueeze(-2)-2*torch::matmul(q,key.transpose(1,2)))/16.;
      lhs->push_back(r.encrypt(k::flatten_pack(ops.diagonalize(s[a]),ops.diagonalize(s[b])),spec.name=="av"?20:15));
      rhs->push_back(r.encrypt(k::flatten_pack(torch::cat({value[a],value[b]},1)),15));
    }
  }
  auto packed_mask=ops.pack_score_mask(f.mask);
  auto evaluate=[&ops,lhs,rhs,values,output,packed_mask,name=spec.name] {
    output->resize(6);
    for(int i=0;i<6;++i) {
      if(name=="qk_av"){auto* published=dynamic_cast<PublishedOperators*>(&ops);(*output)[i]=published?published->qk_av((*lhs)[i],(*rhs)[i],(*values)[i]):ops.av(ops.qk((*lhs)[i],(*rhs)[i]),(*values)[i]);}
      else if(name=="qk") (*output)[i]=ops.qk((*lhs)[i],(*rhs)[i]);
      else if(name=="gaussian_scores") (*output)[i]=dynamic_cast<ResearchOperators&>(ops).gaussian_scores((*lhs)[i],(*rhs)[i]);
      else if(name=="softmax"||name=="softmax_shifted") (*output)[i]=ops.softmax((*lhs)[i],packed_mask);
      else if(name=="av") (*output)[i]=ops.av((*lhs)[i],(*rhs)[i]);
      else (*output)[i]=ops.softmax_v((*lhs)[i],(*rhs)[i],packed_mask);
    }
  };
  auto decode=[&ops,output,pairs,name=spec.name] {
    std::vector<torch::Tensor> result(12);
    for(int i=0;i<6;++i) {
      auto t=(name=="qk"||name=="gaussian_scores"||name=="softmax"||name=="softmax_shifted")?ops.unpack_qk((*output)[i]):ops.unpack_av((*output)[i]);
      result[pairs[i][0]]=t[0];result[pairs[i][1]]=t[1];
    }
    return torch::stack(result);
  };
  auto gaussian_reference=-(q.square().sum(-1,true)+key.square().sum(-1).unsqueeze(-2)-2*torch::matmul(q,key.transpose(1,2)))/16.;
  auto ref=gaussian?gaussian_reference:spec.name=="qk"?scores:(spec.name=="softmax"||shifted)?probabilities:torch::matmul(spec.name=="qk_av"?scores:probabilities,value);
  return {evaluate,decode,ref};
}
}

int main(int argc,char** argv) {
  std::map<std::string,std::string> args{{"method","nexus"},{"scope","qkv"},{"runs","1"},{"warmup","0"},{"layers","1"},{"seed","2026"},{"output","results.jsonl"},{"track","common"},{"threads","8"},{"trace","0"},{"calibration",""},{"approx-reference",""},{"fhebert-stats",""},{"atlas-profile",""},{"arion-profile",""},{"domain-profile",""}};
  try {
    if(argc==2&&std::string(argv[1])=="--capabilities"){std::cout<<"{\"input_encryption_modes\":[\"public_key\",\"client_symmetric\"],\"unsupported_client_symmetric_methods\":[],\"unsupported_client_symmetric_tracks\":[\"native\"]}\n";return 0;}
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];
      if(key=="--list") {for(auto& s:scopes) std::cout<<s.name<<"\t"<<s.level<<"\t"<<s.unit<<"\n";return 0;}
      if(key=="--help") {std::cout<<"cipheropbench_bench --scope NAME|all --runs 1 --warmup 0 --layers 1|12 --track common|native --fixture DIR --output FILE --trace 0|1\n";return 0;}
      if(key.rfind("--",0)!=0 || i+1==argc) throw std::runtime_error("arguments require --name value");
      key=key.substr(2);
      if(!args.count(key)&&key!="fixture") throw std::runtime_error("unknown argument: "+key);
      args[key]=argv[++i];
    }
    int runs=std::stoi(args["runs"]),warmup=std::stoi(args["warmup"]),layers=std::stoi(args["layers"]),seed=std::stoi(args["seed"]);
    TORCH_CHECK(runs>0 && warmup>=0 && layers>=1 && layers<=12,"invalid run counts or layer count");
    TORCH_CHECK(args["track"]=="common"||args["track"]=="native","invalid track");
    TORCH_CHECK(args["track"]!="native" || args["method"]=="nexus", "native track is the NEXUS release; Euston/THOR support common only");
    TORCH_CHECK(args["track"]=="common"||Runtime::input_encryption_mode()=="public_key","Native track supports public_key inputs only");
    (void)Runtime::input_encryption_mode();
    torch::set_num_threads(std::stoi(args["threads"]));
    torch::NoGradGuard guard;
    auto fixture=args.count("fixture")?load_fixture(args["fixture"],layers):synthetic_fixture(layers,seed);
    std::vector<Scope> selected;
    for(const auto& s:scopes) if((args["scope"]=="all"||args["scope"]==s.name)&&(s.name!="gaussian_scores"||args["method"]=="stip")) selected.push_back(s);
    TORCH_CHECK(!selected.empty(),"unknown scope: ",args["scope"]);
    for(const auto& s:selected)TORCH_CHECK(extended_scope_supported(args["method"],s.name)&&component_scope_supported(args["method"],s.name)&&published_scope_supported(args["method"],s.name),"unsupported FHE-only scope for ",args["method"],": ",s.name);
    bool boot=std::any_of(selected.begin(),selected.end(),[](const Scope& s){return s.bootstrap;});
    auto start=std::chrono::steady_clock::now();
    bool literature=is_column_method(args["method"])||is_research_method(args["method"])||args["method"]=="euston"||args["method"]=="thor"||args["method"]=="blb"||args["method"]=="fhe_bert_tiny";
    bool low_head=std::any_of(selected.begin(),selected.end(),[](const Scope& s){return s.name=="qk"||s.name=="av";});
    TORCH_CHECK(!(is_research_method(args["method"])||args["method"]=="arion")||args["calibration"].empty(),"Use the research method profile, not another method calibration");
    TORCH_CHECK((args["method"]!="blb"&&args["method"]!="fhe_bert_tiny")||args["calibration"].empty(),"Existing four-method calibration cannot be silently reused by the new methods");
    auto calibration=calibration::load(args["calibration"],args["method"]);
    TORCH_CHECK(!calibration.enabled||args["track"]=="common","Calibration supports the common graph only");
    Runtime runtime(is_published_method(args["method"])||is_component_method(args["method"])||is_column_method(args["method"])||is_research_method(args["method"])||boot||(literature&&low_head)||args["method"]=="blb"||args["method"]=="fhe_bert_tiny");
    runtime.calibration=std::move(calibration);
    auto setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    runtime.layer_completed=[](size_t completed,size_t count){std::cerr<<"encoder layer "<<completed<<'/'<<count<<" completed"<<std::endl;};
    register_nexus_method();
    TORCH_CHECK(args["arion-profile"].empty()||args["method"]=="arion","ARION profile applies to ARION only");
    register_column_methods(args["arion-profile"]);
    register_component_methods();
    register_published_methods();
    const bool published_domains=args["domain-profile"]=="paper-domains-v1";
    TORCH_CHECK(args["domain-profile"].empty()||published_domains,"Unknown published-domain profile");
    TORCH_CHECK(!published_domains||((args["method"]=="euston"||args["method"]=="stip"||args["method"]=="thor")&&args["calibration"].empty()),"Published-domain profile requires uncalibrated Euston, STIP or THOR");
    TORCH_CHECK(!published_domains||args["method"]!="thor"||args["scope"]=="softmax_shifted","THOR domain profile requires supplied row-max-shifted scores");
    register_literature_methods(published_domains);
    register_extended_methods(args["fhebert-stats"]);
    register_research_methods(args["atlas-profile"],published_domains);
    auto implementation=MethodRegistry::instance().create(args["method"],runtime);
    auto& ops=*implementation;
    const auto& descriptor=MethodRegistry::instance().descriptor(args["method"]);
    std::ofstream stream(args["output"],std::ios::app);
    TORCH_CHECK(stream.good(),"cannot write results: ",args["output"]);
    cudaDeviceProp device{};int device_id;cudaGetDevice(&device_id);cudaGetDeviceProperties(&device,device_id);
    int failures=0;
    for(const auto& scope:selected) {
      try {
        std::string comparison_group=is_published_method(args["method"])?(args["method"]=="cgf_softmax"?"method_specific_nonlinear_variant":"same_target_operator_common_weights"):(is_column_method(args["method"])||is_component_method(args["method"]))?"same_target_operator_common_weights":is_research_method(args["method"])?((args["method"]=="stip"||args["method"]=="ensi")&&scope.name!="linear"&&scope.name!="qkv"&&scope.name!="qk"&&scope.name!="av"&&scope.level!="primitive"?"method_specific_nonlinear_variant":"same_target_operator_common_weights"):"existing_method_contract";
        std::string reference_kind=scope.name=="gaussian_scores"?"exact_gaussian_scores":scope.name=="qk_av"?"exact_matrix_chain_no_softmax":args["track"]=="common"?"exact_transformer":"native_release_plaintext";
        auto test=make_case(scope,runtime,ops,fixture,args["track"],args["method"]);
        runtime.tracing=false;
        // Check one evaluation before spending time on warmup or repeated timing.
        test.evaluate();runtime.synchronize();
        auto actual=test.decode();
        TORCH_CHECK(actual.sizes()==test.reference.sizes(),"output contract mismatch");
        auto reference=test.reference;
        if(args["track"]=="common"&&scope.name!="bootstrap"&&scope.name!="ckks_add"&&scope.name!="ckks_mul") {
          auto valid=fixture.mask[0].unsqueeze(-1);
          if(actual.dim()==3)valid=valid.unsqueeze(0);
          valid=valid.expand_as(actual);
          if(scope.name=="qk"||scope.name=="gaussian_scores"||scope.name=="softmax"||scope.name=="softmax_shifted")valid=valid & fixture.mask.unsqueeze(0);
          actual=actual.masked_select(valid);reference=reference.masked_select(valid);
        }
        bool finite=torch::isfinite(actual).all().item<bool>();
        auto diff=(actual-reference).abs();
        double mae=diff.mean().item<double>(),max_error=diff.max().item<double>(),rmse=(diff.square().mean().sqrt()).item<double>();
        double relative_l2=(actual-reference).norm().item<double>()/std::max(reference.norm().item<double>(),1e-30);
        bool close=finite&&torch::allclose(actual,reference,1e-3,1e-2);
        double ckks_mae=mae,ckks_max=max_error;bool ckks_close=close;auto ckks_reference=reference;
        if(!args["approx-reference"].empty()){
          TORCH_CHECK(runtime.calibration.enabled||is_column_method(args["method"])||is_research_method(args["method"])||args["method"]=="fhe_bert_tiny"||args["method"]=="cgf_softmax"||args["method"]=="nexus"||args["method"]=="euston"||args["method"]=="thor","Approximate reference requires an explicit method circuit");
          auto values=calibration::read_reference(args["approx-reference"],actual.numel());
          auto expected=torch::from_blob(values.data(),{actual.numel()},torch::kDouble).clone();
          ckks_reference=expected;auto error=(actual.reshape({-1})-expected).abs();ckks_mae=error.mean().item<double>();ckks_max=error.max().item<double>();
          ckks_close=finite&&torch::allclose(actual.reshape({-1}),expected,1e-3,1e-2);
        }
        // Preserve the FHE correctness gate; approximation fidelity is reported separately.
        bool method_circuit_comparison=!args["approx-reference"].empty();
        if(!cipheropbench::detail::can_time_method(finite,ckks_close,close,method_circuit_comparison)){
          stream<<"{\"schema_version\":1,\"method\":"<<quote(args["method"])<<",\"input_encryption\":"<<quote(Runtime::input_encryption_mode())<<",\"scope\":"<<quote(scope.name)<<",\"arion_profile\":"<<quote(args["arion-profile"])<<",\"published_domain_profile\":"<<quote(args["domain-profile"])<<",\"track\":"<<quote(args["track"])
            <<",\"status\":"<<quote(finite?"numerical_mismatch":"non_finite")<<",\"execution_status\":\"completed\",\"timing_status\":\"skipped_correctness_failed\",\"evidence\":"<<quote(fixture.evidence)
            <<",\"comparison_group\":"<<quote(comparison_group)<<",\"reference_kind\":"<<quote(reference_kind)<<",\"source_revision\":"<<quote(descriptor.source_revision)<<",\"packing_contract\":"<<quote(descriptor.packing_contract)<<",\"ckks_status\":"<<quote(!finite?"non_finite":ckks_close?"validated":"numerical_mismatch")<<",\"mae\":";number(stream,mae);stream<<",\"max_abs_error\":";number(stream,max_error);
          stream<<",\"ckks_mae\":";number(stream,ckks_mae);stream<<",\"ckks_max_abs_error\":";number(stream,ckks_max);stream<<"}\n";stream.flush();
          std::cerr<<scope.name<<": correctness failed; warmup and timing skipped"<<std::endl;++failures;continue;
        }
        for(int i=0;i<warmup;++i) {test.evaluate();runtime.synchronize();}
        std::vector<double> samples;
        // Detailed primitive tracing is a separate pass so synchronization at
        // every event cannot contaminate the measured server latency.
        for(int i=0;i<runs;++i) {
          runtime.synchronize();auto begin=std::chrono::steady_clock::now();
          test.evaluate();runtime.synchronize();
          samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
          // Client output validation stays outside the synchronized server timer.
          auto checked=test.decode();
          if(args["track"]=="common"&&scope.name!="bootstrap"&&scope.name!="ckks_add"&&scope.name!="ckks_mul"){
            auto valid=fixture.mask[0].unsqueeze(-1);if(checked.dim()==3)valid=valid.unsqueeze(0);valid=valid.expand_as(checked);
            if(scope.name=="qk"||scope.name=="gaussian_scores"||scope.name=="softmax"||scope.name=="softmax_shifted")valid=valid & fixture.mask.unsqueeze(0);
            checked=checked.masked_select(valid);
          }
          TORCH_CHECK(torch::isfinite(checked).all().item<bool>()&&torch::allclose(checked.reshape({-1}),ckks_reference.reshape({-1}),1e-3,1e-2),"Repeated-run FHE correctness check failed");
        }
        auto sorted=samples;std::sort(sorted.begin(),sorted.end());
        double median=sorted.size()%2?sorted[sorted.size()/2]:(sorted[sorted.size()/2-1]+sorted[sorted.size()/2])/2.;
        runtime.events.clear();
        if(args["trace"]=="1") {runtime.tracing=true;test.evaluate();runtime.synchronize();runtime.tracing=false;}
        stream<<"{\"schema_version\":1,\"method\":"<<quote(args["method"])<<",\"input_encryption\":"<<quote(Runtime::input_encryption_mode())<<",\"track\":"<<quote(args["track"])<<",\"scope\":"<<quote(scope.name)<<",\"arion_profile\":"<<quote(args["arion-profile"])<<",\"published_domain_profile\":"<<quote(args["domain-profile"])<<",\"hierarchy\":"<<quote(scope.level)<<",\"unit\":"<<quote(scope.unit)
          <<",\"timing_validation_reference\":"<<quote(method_circuit_comparison?"noiseless_method_circuit":"exact_and_optional_noiseless_reference")<<",\"comparison_group\":"<<quote(comparison_group)
          <<",\"evidence\":"<<quote(fixture.evidence)<<",\"status\":"<<quote(!finite?"non_finite":close?"validated":"numerical_mismatch")
          <<",\"calibration_enabled\":"<<(args["calibration"].empty()?"false":"true")<<",\"allocator\":"<<quote(std::getenv("CIPHEROPBENCH_GPU_ARENA")?"persistent_cuda_ipc_event_reuse":"cuda_async_pool")<<",\"packing_contract\":"<<quote(descriptor.packing_contract)<<",\"source_revision\":"<<quote(descriptor.source_revision)<<",\"implementation_track\":"<<quote(is_published_method(args["method"])||is_column_method(args["method"])||is_research_method(args["method"])||args["method"]=="euston"||args["method"]=="thor"||args["method"]=="blb"||args["method"]=="fhe_bert_tiny"?"common_backend_algorithm_port":"existing_method")<<",\"arithmetic_profile\":"<<quote(is_published_method(args["method"])?(args["method"]=="cgf_softmax"?"cgf_order2_ls_exp15_k4_common_bert_no_kd":args["method"]+"_matrix_only_common_backend"):(is_column_method(args["method"])||is_component_method(args["method"]))?(runtime.calibration.enabled?"public_checkpoint_calibrated_"+args["method"]:args["method"]+"_release_circuit_common_backend"):is_research_method(args["method"])?args["method"]+"_published_circuit_common_graph_port":runtime.calibration.enabled?"public_checkpoint_calibrated_"+args["method"]:args["method"]=="blb"?"blb_fhe_matrix_only":args["method"]=="fhe_bert_tiny"?"fhebert_source_two_stage_extended_to_base":args["method"]=="thor"?"thor_release_canonical_units":args["method"]=="euston"?"euston_release_circuits_common_affine":"existing_method")<<",\"reference_kind\":"<<quote(reference_kind)<<",\"active_samples\":1,\"physical_batch_capacity\":1,\"executed_layers\":"<<(scope.name=="encoder"?layers:1)<<",\"backend\":\"PhantomBootstrapping\",\"rotation_policy\":\"naf_tree_power_of_two_keys\",\"gpu\":"<<quote(device.name)<<",\"N\":65536,\"log_scale\":40,\"coeff_modulus_bits\":[51,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,51,51,51,51,51,51,51,51,51,51,51,51,51,51,51]"
          <<",\"source_batch_capacity\":"<<(is_column_method(args["method"])?256:args["method"]=="blockformer"?4:1)<<",\"replicated_logical_input_copies\":"<<(is_column_method(args["method"])?256:args["method"]=="blockformer"?4:1)<<",\"native_layout_between_qk_and_av\":"<<((args["method"]=="blockformer"&&scope.name=="qk_av")?"true":"false")<<",\"layout_adapter_included\":"<<((is_column_method(args["method"])||is_published_method(args["method"])||is_research_method(args["method"]))?"true":"false")<<",\"security\":\"artifact_profile_not_independently_estimated\",\"weight_secrecy\":\"plaintext\",\"input_secrecy\":\"ciphertext\",\"sequence_length\":128,\"hidden_size\":768,\"heads\":12,\"layers\":"<<layers
          <<",\"seed\":"<<seed<<",\"warmup\":"<<warmup<<",\"runs\":"<<runs<<",\"setup_ms\":"<<setup_ms
          <<",\"timing_boundary\":\"server_evaluation_including_required_refresh_and_online_encoding\",\"median_ms\":"<<median<<",\"samples_ms\":[";
        for(size_t i=0;i<samples.size();++i){if(i)stream<<',';number(stream,samples[i]);}
        stream<<"],\"mae\":";number(stream,mae);stream<<",\"rmse\":";number(stream,rmse);stream<<",\"max_abs_error\":";number(stream,max_error);
        stream<<",\"relative_l2\":";number(stream,relative_l2);stream<<",\"reference_min\":";number(stream,test.reference.min().item<double>());stream<<",\"reference_max\":";number(stream,test.reference.max().item<double>());
        stream<<",\"ckks_reference_kind\":"<<quote(args["approx-reference"].empty()?reference_kind:"noiseless_method_circuit")<<",\"ckks_status\":"<<quote(!finite?"non_finite":ckks_close?"validated":"numerical_mismatch");
        stream<<",\"ckks_mae\":";number(stream,ckks_mae);stream<<",\"ckks_max_abs_error\":";number(stream,ckks_max);
        stream<<",\"timed_output_checks\":"<<runs<<",\"refresh_adapter\":"<<quote(std::getenv("CIPHEROPBENCH_LEGACY_PRECISION_REFRESH")?"encrypted_two_pass_residual_gain256":"recorded_backend_refresh")
          <<",\"arion_exp_equivalent_basis\":"<<((std::getenv("CIPHEROPBENCH_ARION_EXP_REBASE")&&std::string(std::getenv("CIPHEROPBENCH_ARION_EXP_REBASE"))=="1")?"true":"false")
          <<",\"low_level_guard\":"<<((std::getenv("CIPHEROPBENCH_LEGACY_GUARD_LOW_LEVELS")&&std::string(std::getenv("CIPHEROPBENCH_LEGACY_GUARD_LOW_LEVELS"))=="1")?"true":"false")
          <<",\"attention_input_kind\":"<<quote(scope.name=="softmax_shifted"?"supplied_row_max_shifted_scores":args["method"]=="stip"&&scope.name=="softmax"?"negative_squared_distance_over16":"standard_scope_input")
          <<",\"allclose_rtol\":0.001,\"allclose_atol\":0.01,\"events\":[";
        for(size_t i=0;i<runtime.events.size();++i) {auto& e=runtime.events[i];if(i)stream<<',';stream<<"{\"name\":"<<quote(e.name)<<",\"ms\":"<<e.milliseconds<<",\"input_chain\":"<<e.input_chain<<",\"output_chain\":"<<e.output_chain<<'}';}
        stream<<"]}\n";stream.flush();
        std::cout<<scope.name<<": "<<median<<" ms; MAE="<<mae<<"; max="<<max_error<<"; "<<(close?"validated":"numerical_mismatch")<<std::endl;
        if(!finite||!close||!ckks_close) ++failures;
      } catch(const std::exception& error) {
        stream<<"{\"schema_version\":1,\"method\":"<<quote(args["method"])<<",\"input_encryption\":"<<quote(Runtime::input_encryption_mode())<<",\"scope\":"<<quote(scope.name)<<",\"arion_profile\":"<<quote(args["arion-profile"])<<",\"published_domain_profile\":"<<quote(args["domain-profile"])<<",\"track\":"<<quote(args["track"])<<",\"status\":\"failed\",\"error\":"<<quote(error.what())<<"}\n";stream.flush();
        std::cerr<<scope.name<<": "<<error.what()<<std::endl;++failures;
      }
    }
    return failures?2:0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<std::endl;return 1;}
}
